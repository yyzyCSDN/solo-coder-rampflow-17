from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import RLock
from typing import Iterable, Mapping

from .stand_orchestration import StandAssignment, StandConflict, StandDemand, StandProfile
from .timeutil import Window, utc


KEEP_STAND = "keep-stand"
REASSIGN_STAND = "reassign-stand"
ASSIGN_STAND = "assign-stand"


class ServicePackageConflict(RuntimeError):
    """Raised when no stand and service combination is feasible."""


class StaleServicePlan(RuntimeError):
    """Raised when a package was planned against an older coordinator generation."""


@dataclass(frozen=True)
class StandServiceSpec:
    task_id: str
    duration_min: int
    earliest_offset_min: int = 0
    latest_finish_offset_min: int | None = None
    preferred_offset_min: int | None = None
    predecessors: frozenset[str] = frozenset()
    resource_options: tuple[str, ...] = ()
    exclusive_group: str | None = None
    adjacent_group: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task id required")
        if self.duration_min <= 0:
            raise ValueError("service duration must be positive")
        if self.latest_finish_offset_min is not None:
            if self.latest_finish_offset_min < self.earliest_offset_min:
                raise ValueError("latest finish precedes earliest start")
            if self.latest_finish_offset_min < self.duration_min:
                raise ValueError("service cannot finish in its allowed window")
        preferred = self.earliest_offset_min if self.preferred_offset_min is None else self.preferred_offset_min
        if preferred < self.earliest_offset_min:
            raise ValueError("preferred service time precedes earliest time")
        if self.task_id in self.predecessors:
            raise ValueError("service cannot precede itself")
        object.__setattr__(self, "preferred_offset_min", preferred)


@dataclass(frozen=True)
class StandServiceRequest:
    demand: StandDemand
    services: tuple[StandServiceSpec, ...]
    preferred_stand: str | None = None
    candidate_stands: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if not self.services:
            raise ValueError("at least one stand service required")
        ids = [s.task_id for s in self.services]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate service task id")
        known = set(ids)
        for spec in self.services:
            missing = spec.predecessors - known
            if missing:
                raise ValueError(f"unknown predecessors for {spec.task_id}: {sorted(missing)}")


@dataclass(frozen=True)
class _TimeBounds:
    start: datetime
    end: datetime


@dataclass(frozen=True)
class ScheduledStandService:
    flight_id: str
    task_id: str
    stand_id: str
    window: Window
    resource_id: str | None
    exclusive_group: str | None
    adjacent_group: str | None


@dataclass(frozen=True)
class StandServicePlan:
    request: StandServiceRequest
    assignment: StandAssignment
    services: Mapping[str, ScheduledStandService]
    strategy: str
    adjustment_min: int
    generation: int


class StandServiceCoordinator:
    """Joint stand/support-service planner with all-or-nothing commits."""

    def __init__(self, profiles: Iterable[StandProfile]):
        rows = tuple(profiles)
        self.profiles = {p.stand_id: p for p in rows}
        if not self.profiles:
            raise ValueError("at least one stand required")
        if len(self.profiles) != len(rows):
            raise ValueError("duplicate stand")
        self._lock = RLock()
        self._assignments: dict[str, StandAssignment] = {}
        self._by_stand: dict[str, list[StandAssignment]] = {sid: [] for sid in self.profiles}
        self._services: list[ScheduledStandService] = []
        self._packages: dict[str, StandServicePlan] = {}
        self.generation = 0

    def package_for(self, flight_id: str) -> StandServicePlan | None:
        return self._packages.get(flight_id)

    def stand_for(self, flight_id: str) -> str | None:
        assignment = self._assignments.get(flight_id)
        return None if assignment is None else assignment.stand_id

    def services_for(self, flight_id: str) -> tuple[ScheduledStandService, ...]:
        return tuple(
            sorted(
                (s for s in self._services if s.flight_id == flight_id),
                key=lambda s: (s.window.start, s.task_id),
            )
        )

    def _protected(self, demand: StandDemand) -> Window:
        return Window(
            demand.window.start - timedelta(minutes=demand.buffer_before_min),
            demand.window.end + timedelta(minutes=demand.buffer_after_min),
        )

    def _profile_ok(self, profile: StandProfile, demand: StandDemand) -> bool:
        if demand.wingspan_m > profile.max_wingspan_m:
            return False
        if demand.international and not profile.international_capable:
            return False
        if demand.needs_jetbridge and profile.jetbridge_count < 1:
            return False
        return True

    def _adjacent(self, left: str, right: str) -> bool:
        if left == right:
            return False
        return right in self.profiles[left].adjacent or left in self.profiles[right].adjacent

    def _stand_free(
        self,
        stand_id: str,
        protected: Window,
        ignore_flight: str | None = None,
    ) -> bool:
        for assignment in self._by_stand[stand_id]:
            if assignment.flight_id != ignore_flight and assignment.protected_window.overlaps(protected):
                return False
        for other_id in self.profiles[stand_id].incompatible_simultaneous:
            for assignment in self._by_stand.get(other_id, ()):
                if assignment.flight_id != ignore_flight and assignment.protected_window.overlaps(protected):
                    return False
        return True

    def _stand_candidates(self, request: StandServiceRequest) -> tuple[StandAssignment, ...]:
        demand = request.demand
        preference_rank = {
            terminal: rank for rank, terminal in enumerate(demand.terminal_preferences)
        }
        protected = self._protected(demand)
        candidates = []
        for stand_id, profile in self.profiles.items():
            if request.candidate_stands is not None and stand_id not in request.candidate_stands:
                continue
            if not self._profile_ok(profile, demand):
                continue
            if not self._stand_free(stand_id, protected):
                continue
            rank = preference_rank.get(profile.terminal, len(preference_rank) + 1)
            score = (rank, max(0, profile.jetbridge_count - 1), stand_id)
            candidates.append(
                StandAssignment(
                    demand.flight_id,
                    stand_id,
                    demand.window,
                    protected,
                    self.generation,
                    score,
                )
            )
        return tuple(sorted(candidates, key=lambda a: a.score))

    def _bounds(
        self,
        request: StandServiceRequest,
        spec: StandServiceSpec,
    ) -> tuple[_TimeBounds, timedelta, Window]:
        anchor = request.demand.window.start
        duration = timedelta(minutes=spec.duration_min)
        start_lower = max(request.demand.window.start, anchor + timedelta(minutes=spec.earliest_offset_min))
        finish_upper = request.demand.window.end
        if spec.latest_finish_offset_min is not None:
            finish_upper = min(
                finish_upper,
                anchor + timedelta(minutes=spec.latest_finish_offset_min),
            )
        start_upper = finish_upper - duration
        if start_upper < start_lower:
            raise ServicePackageConflict(f"service {spec.task_id} has no feasible window")
        preferred = anchor + timedelta(minutes=spec.preferred_offset_min or 0)
        preferred = max(start_lower, min(start_upper, preferred))
        return _TimeBounds(start_lower, start_upper), duration, Window(preferred, preferred + duration)

    def _applies(
        self,
        spec: StandServiceSpec,
        resource_id: str | None,
        stand_id: str,
        booking: ScheduledStandService,
        window: Window,
    ) -> bool:
        if not booking.window.overlaps(window):
            return False
        if resource_id is not None and booking.resource_id == resource_id:
            return True
        if spec.exclusive_group is not None and booking.exclusive_group == spec.exclusive_group:
            return True
        return (
            spec.adjacent_group is not None
            and booking.adjacent_group == spec.adjacent_group
            and self._adjacent(stand_id, booking.stand_id)
        )

    def _conflicts(
        self,
        spec: StandServiceSpec,
        resource_id: str | None,
        stand_id: str,
        window: Window,
        flight_id: str,
        tentative: tuple[ScheduledStandService, ...],
    ) -> bool:
        for booking in (*self._services, *tentative):
            if booking.flight_id == flight_id:
                # New bookings for the same flight are represented in tentative.
                continue
            if self._applies(spec, resource_id, stand_id, booking, window):
                return True
        for booking in tentative:
            if self._applies(spec, resource_id, stand_id, booking, window):
                return True
        return False

    def _placements(
        self,
        request: StandServiceRequest,
        spec: StandServiceSpec,
        stand_id: str,
        tentative: tuple[ScheduledStandService, ...],
        completion: Mapping[str, Window],
    ) -> tuple[tuple[int, Window, str | None], ...]:
        bounds, duration, preferred = self._bounds(request, spec)
        lower = bounds.start
        for predecessor in spec.predecessors:
            lower = max(lower, completion[predecessor].end)
        upper = bounds.end
        if lower > upper:
            return ()

        resources = spec.resource_options or (None,)
        probe = Window(lower, upper + duration) if lower < upper + duration else None
        placements = []
        seen = set()
        for resource_id in resources:
            candidate_points = {lower, upper, max(lower, min(upper, preferred.start))}
            for booking in (*self._services, *tentative):
                if booking.flight_id == request.demand.flight_id and booking not in tentative:
                    continue
                if probe is not None and not self._applies(spec, resource_id, stand_id, booking, probe):
                    continue
                candidate_points.add(max(lower, min(upper, booking.window.end)))
                before = booking.window.start - duration
                if lower <= before <= upper:
                    candidate_points.add(before)

            for start in sorted(candidate_points):
                key = (start, resource_id)
                if key in seen:
                    continue
                seen.add(key)
                window = Window(start, start + duration)
                if self._conflicts(
                    spec,
                    resource_id,
                    stand_id,
                    window,
                    request.demand.flight_id,
                    tentative,
                ):
                    continue
                adjustment = abs(int((window.start - preferred.start).total_seconds() // 60))
                placements.append((adjustment, window, resource_id))
        return tuple(sorted(placements, key=lambda row: (row[0], row[1].start, "" if row[2] is None else row[2])))

    def _schedule_for_stand(
        self,
        request: StandServiceRequest,
        stand_id: str,
    ) -> tuple[Mapping[str, ScheduledStandService], int] | None:
        specs = {s.task_id: s for s in request.services}
        order = _topological_order(request)
        best: tuple[int, tuple[tuple[str, Window, str | None], ...], dict[str, ScheduledStandService]] | None = None
        nodes = 0

        def search(
            done: set[str],
            tentative: tuple[ScheduledStandService, ...],
            completion: dict[str, ScheduledStandService],
            cost: int,
        ) -> None:
            nonlocal best, nodes
            nodes += 1
            if nodes > 20_000:
                raise ServicePackageConflict("service scheduling search limit exceeded")
            if best is not None and cost > best[0]:
                return
            if len(done) == len(specs):
                ordering = tuple(
                    sorted(
                        (
                            (task_id, item.window, item.resource_id)
                            for task_id, item in completion.items()
                        ),
                        key=lambda row: (row[1].start, row[0]),
                    )
                )
                candidate = (cost, ordering, dict(completion))
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
                return

            task_id = next(item for item in order if item not in done)
            placements = self._placements(
                request,
                specs[task_id],
                stand_id,
                tentative,
                {key: value.window for key, value in completion.items()},
            )
            for adjustment, window, resource_id in placements:
                scheduled = ScheduledStandService(
                    request.demand.flight_id,
                    task_id,
                    stand_id,
                    window,
                    resource_id,
                    specs[task_id].exclusive_group,
                    specs[task_id].adjacent_group,
                )
                next_done = set(done)
                next_done.add(task_id)
                next_completion = dict(completion)
                next_completion[task_id] = scheduled
                search(next_done, tentative + (scheduled,), next_completion, cost + adjustment)

        search(set(), (), {}, 0)
        if best is None:
            return None
        return dict(best[2]), best[0]

    def plan(self, request: StandServiceRequest) -> StandServicePlan:
        normalized_demand = StandDemand(
            request.demand.flight_id,
            Window(utc(request.demand.window.start), utc(request.demand.window.end)),
            request.demand.wingspan_m,
            request.demand.international,
            request.demand.terminal_preferences,
            request.demand.needs_jetbridge,
            request.demand.buffer_before_min,
            request.demand.buffer_after_min,
            request.demand.priority,
        )
        request = StandServiceRequest(
            normalized_demand,
            request.services,
            request.preferred_stand,
            request.candidate_stands,
        )
        with self._lock:
            explicit_preferred = request.preferred_stand is not None
            if explicit_preferred and request.preferred_stand not in self.profiles:
                raise ServicePackageConflict(f"unknown preferred stand {request.preferred_stand}")
            if request.candidate_stands is not None:
                unknown = request.candidate_stands - set(self.profiles)
                if unknown:
                    raise ServicePackageConflict(f"unknown candidate stands: {sorted(unknown)}")
            candidates = self._stand_candidates(request)
            if not candidates:
                raise ServicePackageConflict(f"no feasible stand for {request.demand.flight_id}")

            preferred = None
            if explicit_preferred:
                preferred = next((a for a in candidates if a.stand_id == request.preferred_stand), None)
                if preferred is not None:
                    scheduled = self._schedule_for_stand(request, preferred.stand_id)
                    if scheduled is not None:
                        services, adjustment = scheduled
                        return StandServicePlan(
                            request, preferred, services, KEEP_STAND, adjustment, self.generation
                        )
                # A stand that fails the joint stand/service check is not a
                # keepable option; fall through to full reassignment.
                preferred = None

            alternatives = []
            for assignment in candidates:
                if preferred is not None and assignment.stand_id == preferred.stand_id:
                    continue
                scheduled = self._schedule_for_stand(request, assignment.stand_id)
                if scheduled is None:
                    continue
                services, adjustment = scheduled
                alternatives.append((adjustment, assignment.score, assignment.stand_id, services, assignment))

            if not alternatives:
                raise ServicePackageConflict(
                    f"no jointly feasible stand service package for {request.demand.flight_id}"
                )
            adjustment, _, _, services, assignment = min(
                alternatives,
                key=lambda item: (item[0], item[1], item[2]),
            )
            strategy = REASSIGN_STAND if explicit_preferred else ASSIGN_STAND
            return StandServicePlan(
                request, assignment, services, strategy, adjustment, self.generation
            )

    def _validate_plan(
        self,
        plan: StandServicePlan,
        assignments: Mapping[str, StandAssignment],
        by_stand: Mapping[str, list[StandAssignment]],
        services: tuple[ScheduledStandService, ...],
    ) -> None:
        request = plan.request
        demand = request.demand
        assignment = plan.assignment
        if assignment.stand_id not in self.profiles:
            raise StandConflict("unknown stand")
        if not self._profile_ok(self.profiles[assignment.stand_id], demand):
            raise StandConflict("stand no longer compatible")
        if request.candidate_stands is not None and assignment.stand_id not in request.candidate_stands:
            raise StandConflict("stand outside planned candidates")
        if self._protected(demand) != assignment.protected_window:
            raise StandConflict("stand protection window changed")
        for flight_id, existing in assignments.items():
            if flight_id == demand.flight_id:
                if existing != assignment:
                    raise StandConflict("flight already has a different stand")
                continue
            if existing.stand_id == assignment.stand_id and existing.protected_window.overlaps(
                assignment.protected_window
            ):
                raise StandConflict("stand became unavailable")
        for other_id in self.profiles[assignment.stand_id].incompatible_simultaneous:
            for existing in by_stand.get(other_id, ()):
                if existing.flight_id != demand.flight_id and existing.protected_window.overlaps(
                    assignment.protected_window
                ):
                    raise StandConflict("simultaneous stand restriction conflict")

        specs = {s.task_id: s for s in request.services}
        if set(plan.services) != set(specs):
            raise ServicePackageConflict("planned services do not match request")
        tentative: list[ScheduledStandService] = []
        completion: dict[str, Window] = {}
        anchor = demand.window.start
        order = _topological_order(request)
        for task_id in order:
            spec = specs[task_id]
            scheduled = plan.services[task_id]
            if scheduled.flight_id != demand.flight_id or scheduled.stand_id != assignment.stand_id:
                raise ServicePackageConflict(f"invalid planned service {task_id}")
            if scheduled.window.minutes != spec.duration_min:
                raise ServicePackageConflict(f"service {task_id} duration changed")
            if scheduled.resource_id is not None and scheduled.resource_id not in spec.resource_options:
                raise ServicePackageConflict(f"invalid resource for {task_id}")
            if scheduled.exclusive_group != spec.exclusive_group:
                raise ServicePackageConflict(f"exclusive group mismatch for {task_id}")
            if scheduled.adjacent_group != spec.adjacent_group:
                raise ServicePackageConflict(f"adjacent group mismatch for {task_id}")
            window = scheduled.window
            if window.start < demand.window.start or window.end > demand.window.end:
                raise ServicePackageConflict(f"service {task_id} outside stand occupancy")
            if window.start < anchor + timedelta(minutes=spec.earliest_offset_min):
                raise ServicePackageConflict(f"service {task_id} starts too early")
            if spec.latest_finish_offset_min is not None and window.end > anchor + timedelta(
                minutes=spec.latest_finish_offset_min
            ):
                raise ServicePackageConflict(f"service {task_id} misses deadline")
            for predecessor in spec.predecessors:
                if window.start < completion[predecessor].end:
                    raise ServicePackageConflict(f"service {task_id} violates predecessor {predecessor}")
            for existing in (*services, *tentative):
                if self._applies(
                    spec,
                    scheduled.resource_id,
                    assignment.stand_id,
                    existing,
                    window,
                ):
                    if existing.flight_id == demand.flight_id and existing not in tentative:
                        continue
                    raise ServicePackageConflict(f"service conflict for {task_id}")
            tentative.append(scheduled)
            completion[task_id] = window

    def commit(self, plan: StandServicePlan) -> StandServicePlan:
        with self._lock:
            if plan.generation != self.generation:
                raise StaleServicePlan(
                    f"planned at generation {plan.generation}, current generation {self.generation}"
                )
            flight_id = plan.request.demand.flight_id
            existing = self._packages.get(flight_id)
            if existing is not None:
                if existing == plan:
                    return existing
                raise ServicePackageConflict("flight already has a committed service package")

            # Validate against shadows first; the commit phase performs no checks
            # that can fail after coordinator state has started to change.
            self._validate_plan(
                plan,
                dict(self._assignments),
                {sid: list(rows) for sid, rows in self._by_stand.items()},
                tuple(self._services),
            )

            new_assignment = _assignment_with_generation(plan.assignment, self.generation + 1)
            self._assignments[flight_id] = new_assignment
            self._by_stand[new_assignment.stand_id].append(new_assignment)
            self._by_stand[new_assignment.stand_id].sort(
                key=lambda a: (a.protected_window.start, a.flight_id)
            )
            new_services = {}
            for task_id in _topological_order(plan.request):
                scheduled = plan.services[task_id]
                self._services.append(scheduled)
                new_services[task_id] = scheduled
            self._services.sort(key=lambda s: (s.window.start, s.stand_id, s.task_id))
            committed = StandServicePlan(
                plan.request,
                new_assignment,
                new_services,
                plan.strategy,
                plan.adjustment_min,
                self.generation + 1,
            )
            self._packages[flight_id] = committed
            self.generation += 1
            return committed

    def release(self, flight_id: str) -> bool:
        with self._lock:
            plan = self._packages.pop(flight_id, None)
            if plan is None:
                return False
            assignment = self._assignments.pop(flight_id)
            self._by_stand[assignment.stand_id] = [
                item for item in self._by_stand[assignment.stand_id] if item.flight_id != flight_id
            ]
            self._services = [item for item in self._services if item.flight_id != flight_id]
            self.generation += 1
            return True


def _topological_order(request: StandServiceRequest) -> tuple[str, ...]:
    remaining = {s.task_id: set(s.predecessors) for s in request.services}
    ordered = []
    while remaining:
        ready = sorted(task_id for task_id, predecessors in remaining.items() if not predecessors)
        if not ready:
            raise ValueError("service dependency cycle")
        for task_id in ready[:1]:
            ordered.append(task_id)
            del remaining[task_id]
            for predecessors in remaining.values():
                predecessors.discard(task_id)
    return tuple(ordered)


def _assignment_with_generation(assignment: StandAssignment, generation: int) -> StandAssignment:
    return StandAssignment(
        assignment.flight_id,
        assignment.stand_id,
        assignment.window,
        assignment.protected_window,
        generation,
        assignment.score,
    )
