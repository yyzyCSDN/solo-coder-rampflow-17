"""Atomic joint coordination of stand allocation and turnaround services.

Each individual domain (stands, catering, fueling, ...) validates its own
bookings, but a package is only feasible once *cross-domain* constraints are
considered together: an adjacent-stand safety envelope and a same-stand
mutual-exclusion group can each be satisfiable in isolation while the combined
package clashes.

This planner solves the package against a private **shadow** state and only
applies it to live state when every booking succeeded.  Resolution policy:

* ``AUTO`` (default) first tries to keep the stand and shift service times; if
  that is infeasible it replans the whole package on another stand.
* ``KEEP_STAND_SHIFT_SERVICES`` / ``REALLOCATE_STAND`` pin one behaviour.

No partial package can ever be written: ``coordinate`` either commits the stand
and all services atomically or raises and leaves committed state untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Iterable, Mapping, Sequence

from .timeutil import Window, utc

# Bookings without a concrete resource are held in a per-stand virtual bucket;
# they still participate in same-stand mutual-exclusion group checks.
_STAND_BUCKET = "stand:{stand_id}"


class CoordinationStrategy(str, Enum):
    AUTO = "auto"
    KEEP_STAND_SHIFT_SERVICES = "keep-stand-shift-services"
    REALLOCATE_STAND = "reallocate-stand"


@dataclass(frozen=True)
class CoordinationStand:
    """A stand as seen by the joint coordinator.

    ``adjacent`` models the adjacent-stand restriction: the stand may not be
    occupied inside its protected window while any listed neighbouring stand
    is occupied inside theirs.
    """

    stand_id: str
    terminal: str = ""
    protected_before_min: int = 5
    protected_after_min: int = 10
    adjacent: frozenset[str] = frozenset()
    preferred: bool = False

    def protected(self, window: Window) -> Window:
        return window.expand(
            before=self.protected_before_min,
            after=self.protected_after_min,
        )


@dataclass(frozen=True)
class CoordinationService:
    """A single ground-handling service that must fit the turnaround."""

    service: str
    duration_min: int
    resource_id: str | None = None
    # Services sharing an exclusion group may never overlap while on the same
    # stand, even when they use distinct resources (e.g. fueling vs catering).
    exclusion_group: str | None = None
    earliest_offset_min: int = 0
    # Offset from on-block; the service must finish by on-block + this value.
    latest_finish_offset_min: int | None = None
    hard: bool = True
    predecessors: frozenset[str] = frozenset()

    def __post_init__(self):
        if self.duration_min <= 0:
            raise ValueError("service duration must be positive")
        if self.service in self.predecessors:
            raise ValueError("service cannot precede itself")
        if (
            self.latest_finish_offset_min is not None
            and self.latest_finish_offset_min
            < self.earliest_offset_min + self.duration_min
        ):
            raise ValueError("service cannot fit inside its own window")


@dataclass(frozen=True)
class CoordinationRequest:
    flight_id: str
    on_block: datetime
    off_block: datetime
    services: tuple[CoordinationService, ...]
    preferred_stand: str | None = None
    # When set, overrides the coordinator's default strategy for this request.
    strategy: CoordinationStrategy | None = None

    def __post_init__(self):
        on, off = utc(self.on_block), utc(self.off_block)
        if off <= on:
            raise ValueError("off-block must follow on-block")
        object.__setattr__(self, "on_block", on)
        object.__setattr__(self, "off_block", off)
        names = [s.service for s in self.services]
        if len(names) != len(set(names)):
            raise ValueError("duplicate service in coordination request")
        known = set(names)
        for svc in self.services:
            missing = svc.predecessors - known
            if missing:
                raise ValueError(
                    f"unknown predecessors for {svc.service}: {sorted(missing)}"
                )

    @property
    def occupancy(self) -> Window:
        return Window(self.on_block, self.off_block)


@dataclass(frozen=True)
class ServiceBooking:
    service: str
    resource_id: str | None
    window: Window
    shifted: bool = False


@dataclass(frozen=True)
class CoordinationPlan:
    flight_id: str
    stand_id: str
    stand_window: Window
    protected_window: Window
    bookings: Mapping[str, ServiceBooking]
    strategy: CoordinationStrategy
    stand_changed: bool
    notices: tuple[str, ...] = ()

    def booking(self, service: str) -> ServiceBooking:
        return self.bookings[service]


class CoordinationConflict(RuntimeError):
    """No stand/service combination was feasible.

    ``violations`` records every constraint that blocked the attempts so the
    caller can escalate (e.g. extend occupancy, move another flight first).
    """

    def __init__(self, message: str, violations: Sequence[str] = ()):
        self.violations = tuple(violations)
        super().__init__(
            message if not violations else f"{message}: {'; '.join(self.violations)}"
        )


@dataclass(frozen=True)
class _StandHeld:
    flight_id: str
    protected: Window


@dataclass(frozen=True)
class _ServiceHeld:
    flight_id: str
    stand_id: str
    resource_key: str
    window: Window
    exclusion_group: str | None


def _bucket_for(resource_id: str | None, stand_id: str) -> str:
    return resource_id if resource_id is not None else _STAND_BUCKET.format(
        stand_id=stand_id
    )


class StandServiceCoordinator:
    """Joint allocator with shadow-based atomic commits."""

    def __init__(
        self,
        stands: Iterable[CoordinationStand],
        strategy: CoordinationStrategy = CoordinationStrategy.AUTO,
        max_shift_steps: int = 300,
    ):
        stands = tuple(stands)
        self.stands = {s.stand_id: s for s in stands}
        if not self.stands:
            raise ValueError("at least one stand required")
        if len(self.stands) != len(stands):
            raise ValueError("duplicate stand id")
        self.strategy = strategy
        self.max_shift_steps = max_shift_steps
        # Committed state --------------------------------------------------
        self._stands: dict[str, list[_StandHeld]] = {sid: [] for sid in self.stands}
        self._resources: dict[str, list[_ServiceHeld]] = {}
        self._by_flight: dict[str, CoordinationPlan] = {}

    # ------------------------------------------------------------------ views
    def plan_for(self, flight_id: str) -> CoordinationPlan:
        return self._by_flight[flight_id]

    def occupancy(self, stand_id: str) -> tuple[_StandHeld, ...]:
        return tuple(self._stands[stand_id])

    def resource_bookings(self, resource_id: str) -> tuple[_ServiceHeld, ...]:
        return tuple(self._resources.get(resource_id, ()))

    # ------------------------------------------------------------- ordering
    @staticmethod
    def _ordered(services: Sequence[CoordinationService]) -> list[CoordinationService]:
        """Topological order so predecessors are placed first."""
        remaining = {s.service: s for s in services}
        done: set[str] = set()
        order: list[CoordinationService] = []
        while remaining:
            ready = sorted(
                name for name, svc in remaining.items() if svc.predecessors <= done
            )
            if not ready:
                raise ValueError("service dependency cycle")
            for name in ready:
                order.append(remaining.pop(name))
                done.add(name)
        return order

    def _rank_stands(self, request: CoordinationRequest) -> list[str]:
        def key(sid: str) -> tuple:
            p = self.stands[sid]
            return (
                0 if sid == request.preferred_stand else 1,
                0 if p.preferred else 1,
                p.terminal,
                sid,
            )

        return sorted(self.stands, key=key)

    # ------------------------------------------------------------- conflicts
    def _stand_clash(
        self,
        shadow_stands: Mapping[str, list[_StandHeld]],
        stand_id: str,
        protected: Window,
        ignore_flight: str,
    ) -> str | None:
        for held in shadow_stands.get(stand_id, ()):
            if held.flight_id == ignore_flight:
                continue
            if held.protected.overlaps(protected):
                return f"stand {stand_id} occupied by {held.flight_id}"
        # Adjacent-stand restriction in both declaration directions: a stand
        # is restricted not only by its own `adjacent` list but also by any
        # neighbour that lists it (handles asymmetric rule configuration).
        neighbours = set(self.stands[stand_id].adjacent)
        for other_id, other in self.stands.items():
            if other_id != stand_id and stand_id in other.adjacent:
                neighbours.add(other_id)
        for other_id in neighbours:
            if other_id not in self.stands:
                continue
            for held in shadow_stands.get(other_id, ()):
                if held.flight_id == ignore_flight:
                    continue
                if held.protected.overlaps(protected):
                    return (
                        f"adjacent-stand restriction with {other_id} "
                        f"(occupied by {held.flight_id})"
                    )
        return None

    @staticmethod
    def _service_clash(
        all_holds: Sequence[_ServiceHeld],
        stand_id: str,
        window: Window,
        resource_key: str,
        group: str | None,
        ignore_flight: str,
    ) -> str | None:
        for held in all_holds:
            if held.flight_id == ignore_flight or not held.window.overlaps(window):
                continue
            if held.resource_key == resource_key:
                return f"resource {resource_key} used by {held.flight_id}"
            if (
                group is not None
                and held.exclusion_group == group
                and held.stand_id == stand_id
            ):
                return (
                    f"exclusion group '{group}' on stand {stand_id} "
                    f"already active for {held.flight_id}"
                )
        return None

    @staticmethod
    def _index_holds(
        shadow_resources: Mapping[str, list[_ServiceHeld]],
        tentative: Sequence[_ServiceHeld],
    ) -> list[_ServiceHeld]:
        out: list[_ServiceHeld] = []
        for holds in shadow_resources.values():
            out.extend(holds)
        out.extend(tentative)
        return out

    # ------------------------------------------------------------- placement
    def _place_services(
        self,
        shadow_resources: dict[str, list[_ServiceHeld]],
        request: CoordinationRequest,
        stand_id: str,
        stand_window: Window,
    ) -> tuple[dict[str, ServiceBooking], list[str]]:
        """Place every service, using a local tentative layer.

        Nothing is written into ``shadow_resources`` unless every *hard*
        service is placed, so a failed attempt cannot contaminate a later
        attempt on another stand.
        """
        tentative: list[_ServiceHeld] = []
        bookings: dict[str, ServiceBooking] = {}
        soft_violations: list[str] = []

        for svc in self._ordered(request.services):
            earliest = request.on_block + timedelta(minutes=svc.earliest_offset_min)
            if svc.predecessors:
                earliest = max(
                    earliest,
                    max(bookings[p].window.end for p in svc.predecessors),
                )
            deadline = request.off_block
            if svc.latest_finish_offset_min is not None:
                deadline = min(
                    deadline,
                    request.on_block
                    + timedelta(minutes=svc.latest_finish_offset_min),
                )
            duration = timedelta(minutes=svc.duration_min)
            start = earliest
            original_start = earliest
            key = _bucket_for(svc.resource_id, stand_id)
            placed: Window | None = None
            reason = ""
            for _ in range(self.max_shift_steps):
                window = Window(start, start + duration)
                all_holds = self._index_holds(shadow_resources, tentative)
                clash = self._service_clash(
                    all_holds,
                    stand_id,
                    window,
                    key,
                    svc.exclusion_group,
                    # Nothing is ignored: the tentative layer already holds
                    # *this flight's* earlier services, and those must count
                    # (same-flight exclusion group / predecessor serialisation).
                    "",
                )
                if clash is None:
                    if window.end > stand_window.end:
                        reason = (
                            f"service {svc.service} runs past stand occupancy "
                            f"on {stand_id}"
                        )
                        break
                    placed = window
                    break
                start = self._next_clear_start(
                    all_holds,
                    stand_id,
                    window,
                    key,
                    svc.exclusion_group,
                    "",
                )
            if placed is not None and placed.end <= deadline:
                bookings[svc.service] = ServiceBooking(
                    svc.service,
                    svc.resource_id,
                    placed,
                    shifted=placed.start != original_start,
                )
                tentative.append(
                    _ServiceHeld(
                        request.flight_id,
                        stand_id,
                        key,
                        placed,
                        svc.exclusion_group,
                    )
                )
                continue
            if not reason:
                reason = (
                    f"service {svc.service} cannot finish by "
                    f"{deadline:%H:%M} on {stand_id}"
                    if placed is not None
                    else f"service {svc.service} blocked on {stand_id}"
                )
            if svc.hard:
                return {}, [reason]
            soft_violations.append(reason)

        # All hard services fit: publish the tentative layer into the shadow.
        for held in tentative:
            shadow_resources.setdefault(held.resource_key, []).append(held)
        return bookings, soft_violations

    @staticmethod
    def _next_clear_start(
        all_holds: Sequence[_ServiceHeld],
        stand_id: str,
        window: Window,
        resource_key: str,
        group: str | None,
        ignore_flight: str,
    ) -> datetime:
        ends = [window.end]
        for held in all_holds:
            if held.flight_id == ignore_flight or not held.window.overlaps(window):
                continue
            if held.resource_key == resource_key or (
                group is not None
                and held.exclusion_group == group
                and held.stand_id == stand_id
            ):
                ends.append(held.window.end)
        return max(ends)

    # ------------------------------------------------------------- attempts
    def _attempt_on(
        self,
        request: CoordinationRequest,
        stand_id: str,
        strategy_used: CoordinationStrategy,
        shadow_stands: dict[str, list[_StandHeld]],
        shadow_resources: dict[str, list[_ServiceHeld]],
    ) -> tuple[CoordinationPlan | None, list[str]]:
        stand_window = request.occupancy
        protected = self.stands[stand_id].protected(stand_window)
        clash = self._stand_clash(
            shadow_stands, stand_id, protected, request.flight_id
        )
        if clash is not None:
            return None, [clash]

        bookings, soft_violations = self._place_services(
            shadow_resources, request, stand_id, stand_window
        )
        if not bookings:
            return None, soft_violations

        shadow_stands[stand_id].append(_StandHeld(request.flight_id, protected))
        shifted = sorted(b.service for b in bookings.values() if b.shifted)
        notices = list(soft_violations)
        if shifted:
            notices.append("services shifted: " + ", ".join(shifted))
        return (
            CoordinationPlan(
                request.flight_id,
                stand_id,
                stand_window,
                protected,
                bookings,
                strategy_used,
                stand_changed=request.preferred_stand is not None
                and stand_id != request.preferred_stand,
                notices=tuple(notices),
            ),
            [],
        )

    def _fresh_shadow(
        self,
        drop_flight: str | None = None,
    ) -> tuple[dict[str, list[_StandHeld]], dict[str, list[_ServiceHeld]]]:
        stands = {sid: list(holds) for sid, holds in self._stands.items()}
        resources = {rid: list(holds) for rid, holds in self._resources.items()}
        if drop_flight is not None:
            for sid in stands:
                stands[sid] = [
                    h for h in stands[sid] if h.flight_id != drop_flight
                ]
            for rid in list(resources):
                resources[rid] = [
                    h for h in resources[rid] if h.flight_id != drop_flight
                ]
        return stands, resources

    def _solve_on_shadow(
        self,
        request: CoordinationRequest,
        shadow_stands: dict[str, list[_StandHeld]],
        shadow_resources: dict[str, list[_ServiceHeld]],
    ) -> tuple[CoordinationPlan | None, list[str]]:
        strategy = request.strategy or self.strategy
        ranked = self._rank_stands(request)
        preferred = request.preferred_stand or ranked[0]
        if preferred not in self.stands:
            raise CoordinationConflict(f"unknown stand {preferred}")
        violations: list[str] = []

        # Phase 1 (AUTO / KEEP): keep the stand, only move service times.
        if strategy in (
            CoordinationStrategy.AUTO,
            CoordinationStrategy.KEEP_STAND_SHIFT_SERVICES,
        ):
            # A private shadow: its tentative bookings must not leak into the
            # reallocation phase or, on failure, into committed state.
            st, res = self._copy_shadow(shadow_stands, shadow_resources)
            plan, errs = self._attempt_on(
                request,
                preferred,
                CoordinationStrategy.KEEP_STAND_SHIFT_SERVICES,
                st,
                res,
            )
            if plan is not None:
                shadow_stands.clear()
                shadow_stands.update(st)
                shadow_resources.clear()
                shadow_resources.update(res)
                return plan, []
            violations.extend(f"[keep {preferred}] {e}" for e in errs)
            if strategy == CoordinationStrategy.KEEP_STAND_SHIFT_SERVICES:
                return None, violations

        # Phase 2 (AUTO / REALLOCATE): whole package may move to another stand.
        if strategy in (
            CoordinationStrategy.AUTO,
            CoordinationStrategy.REALLOCATE_STAND,
        ):
            for sid in ranked:
                st, res = self._copy_shadow(shadow_stands, shadow_resources)
                plan, errs = self._attempt_on(
                    request,
                    sid,
                    CoordinationStrategy.REALLOCATE_STAND,
                    st,
                    res,
                )
                if plan is not None:
                    shadow_stands.clear()
                    shadow_stands.update(st)
                    shadow_resources.clear()
                    shadow_resources.update(res)
                    return plan, []
                violations.extend(f"[{sid}] {e}" for e in errs)
        return None, violations

    @staticmethod
    def _copy_shadow(
        shadow_stands: Mapping[str, list[_StandHeld]],
        shadow_resources: Mapping[str, list[_ServiceHeld]],
    ) -> tuple[dict[str, list[_StandHeld]], dict[str, list[_ServiceHeld]]]:
        return (
            {sid: list(holds) for sid, holds in shadow_stands.items()},
            {rid: list(holds) for rid, holds in shadow_resources.items()},
        )

    # ------------------------------------------------------------- public API
    def coordinate(self, request: CoordinationRequest) -> CoordinationPlan:
        """Atomically allocate a stand and place every service.

        Raises ``CoordinationConflict`` without mutating committed state when
        the package is infeasible under the configured strategy.
        """
        if request.flight_id in self._by_flight:
            existing = self._by_flight[request.flight_id]
            if (
                request.preferred_stand is None
                or existing.stand_id == request.preferred_stand
            ):
                return existing
            raise CoordinationConflict(
                f"flight {request.flight_id} already coordinated; "
                "use coordinate_replan or release first"
            )

        shadow_stands, shadow_resources = self._fresh_shadow()
        plan, violations = self._solve_on_shadow(
            request, shadow_stands, shadow_resources
        )
        if plan is None:
            raise CoordinationConflict(
                f"no feasible stand/service package for {request.flight_id}",
                violations,
            )
        self._commit(request, plan)
        return plan

    def coordinate_replan(self, request: CoordinationRequest) -> CoordinationPlan:
        """Replan an existing package atomically.

        The old package is removed only inside a shadow; if the new package
        cannot be solved, committed state (and the old package) is retained.
        """
        old = self._by_flight.get(request.flight_id)
        if old is None:
            return self.coordinate(request)

        shadow_stands, shadow_resources = self._fresh_shadow(drop_flight=request.flight_id)
        plan, violations = self._solve_on_shadow(
            request, shadow_stands, shadow_resources
        )
        if plan is None:
            raise CoordinationConflict(
                f"replan infeasible for {request.flight_id}; "
                "existing package retained",
                violations,
            )
        # Atomic swap of committed state.
        self._stands[old.stand_id] = [
            h for h in self._stands[old.stand_id] if h.flight_id != request.flight_id
        ]
        for rid in list(self._resources):
            self._resources[rid] = [
                h for h in self._resources[rid] if h.flight_id != request.flight_id
            ]
        self._by_flight.pop(request.flight_id, None)
        self._commit(request, plan)
        return plan

    def release(self, flight_id: str) -> bool:
        plan = self._by_flight.pop(flight_id, None)
        if plan is None:
            return False
        self._stands[plan.stand_id] = [
            h for h in self._stands[plan.stand_id] if h.flight_id != flight_id
        ]
        for rid in list(self._resources):
            self._resources[rid] = [
                h for h in self._resources[rid] if h.flight_id != flight_id
            ]
        return True

    # ------------------------------------------------------------- commit
    def _commit(self, request: CoordinationRequest, plan: CoordinationPlan) -> None:
        """The only place that mutates committed state; called solely with a
        fully solved plan, so partial packages are impossible."""
        self._stands[plan.stand_id].append(
            _StandHeld(plan.flight_id, plan.protected_window)
        )
        specs = {s.service: s for s in request.services}
        for booking in plan.bookings.values():
            svc = specs[booking.service]
            self._resources.setdefault(
                _bucket_for(booking.resource_id, plan.stand_id), []
            ).append(
                _ServiceHeld(
                    plan.flight_id,
                    plan.stand_id,
                    _bucket_for(booking.resource_id, plan.stand_id),
                    booking.window,
                    svc.exclusion_group,
                )
            )
        self._by_flight[plan.flight_id] = plan
