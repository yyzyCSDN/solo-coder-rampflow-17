import unittest
from datetime import datetime, timezone

from rampflow.stand_orchestration import StandDemand, StandProfile
from rampflow.stand_service_package import (
    KEEP_STAND,
    REASSIGN_STAND,
    StandServiceCoordinator,
    StandServiceRequest,
    StandServiceSpec,
    StaleServicePlan,
)
from rampflow.timeutil import Window
from rampflow.workflows.control import RampWorkflowControl


UTC = timezone.utc
D = lambda h, m=0: datetime(2026, 1, 1, h, m, tzinfo=UTC)


def make_coordinator():
    return StandServiceCoordinator(
        [
            StandProfile(
                "S1",
                "T",
                60,
                True,
                1,
                adjacent=frozenset({"S2"}),
                incompatible_simultaneous=frozenset(),
            ),
            StandProfile(
                "S2",
                "T",
                60,
                True,
                1,
                adjacent=frozenset({"S1"}),
                incompatible_simultaneous=frozenset(),
            ),
            StandProfile("S3", "T", 60, True, 1),
        ]
    )


def services():
    fuel = StandServiceSpec(
        "fuel",
        duration_min=20,
        earliest_offset_min=10,
        latest_finish_offset_min=80,
        preferred_offset_min=10,
        adjacent_group="airside",
    )
    catering = StandServiceSpec(
        "catering",
        duration_min=20,
        earliest_offset_min=10,
        latest_finish_offset_min=80,
        preferred_offset_min=10,
        predecessors=frozenset({"fuel"}),
        adjacent_group="airside",
    )
    return fuel, catering


def request(flight_id="F1", preferred_stand="S1", window=None, latest_finish_offset_min=80):
    fuel, catering = services()
    fuel = StandServiceSpec(
        fuel.task_id,
        fuel.duration_min,
        earliest_offset_min=fuel.earliest_offset_min,
        latest_finish_offset_min=latest_finish_offset_min,
        preferred_offset_min=fuel.preferred_offset_min,
        exclusive_group=fuel.exclusive_group,
        adjacent_group=fuel.adjacent_group,
    )
    catering = StandServiceSpec(
        catering.task_id,
        catering.duration_min,
        earliest_offset_min=catering.earliest_offset_min,
        latest_finish_offset_min=latest_finish_offset_min,
        preferred_offset_min=catering.preferred_offset_min,
        predecessors=catering.predecessors,
        exclusive_group=catering.exclusive_group,
        adjacent_group=catering.adjacent_group,
    )
    return StandServiceRequest(
        StandDemand(flight_id, window or Window(D(10), D(12)), 40, False, buffer_before_min=0, buffer_after_min=0),
        (fuel, catering),
        preferred_stand=preferred_stand,
    )


def commit_existing(coordinator):
    existing_fuel, existing_catering = services()
    existing = StandServiceRequest(
        StandDemand("G1", Window(D(10), D(11)), 40, False, buffer_before_min=0, buffer_after_min=0),
        (
            StandServiceSpec(
                "g-fuel",
                20,
                preferred_offset_min=0,
                latest_finish_offset_min=20,
                adjacent_group="airside",
            ),
            StandServiceSpec(
                "g-catering",
                20,
                preferred_offset_min=20,
                earliest_offset_min=20,
                latest_finish_offset_min=40,
                predecessors=frozenset({"g-fuel"}),
                adjacent_group="airside",
            ),
        ),
        preferred_stand="S2",
    )
    plan = coordinator.plan(existing)
    return coordinator.commit(plan)


class StandServicePackageTests(unittest.TestCase):
    def test_keeps_preferred_stand_by_adjusting_support_times(self):
        coordinator = make_coordinator()
        commit_existing(coordinator)

        plan = coordinator.plan(request(window=Window(D(10), D(13))))
        committed = coordinator.commit(plan)

        self.assertEqual(committed.strategy, KEEP_STAND)
        self.assertEqual(committed.assignment.stand_id, "S1")
        self.assertEqual(committed.services["fuel"].window.start, D(10, 40))
        self.assertEqual(committed.services["catering"].window.start, D(11))
        self.assertGreater(committed.adjustment_min, 0)

    def test_reassigns_when_stand_cannot_support_joint_constraints(self):
        coordinator = make_coordinator()
        commit_existing(coordinator)

        plan = coordinator.plan(request(latest_finish_offset_min=60))
        committed = coordinator.commit(plan)

        self.assertEqual(committed.strategy, REASSIGN_STAND)
        self.assertEqual(committed.assignment.stand_id, "S3")
        self.assertEqual(committed.services["fuel"].window.start, D(10, 10))
        self.assertEqual(committed.services["catering"].window.start, D(10, 30))

    def test_exclusive_group_blocks_even_on_non_adjacent_stand(self):
        coordinator = make_coordinator()
        occupied = StandServiceRequest(
            StandDemand("G1", Window(D(10), D(11)), 40, False, buffer_before_min=0, buffer_after_min=0),
            (
                StandServiceSpec(
                    "fuel",
                    20,
                    preferred_offset_min=10,
                    latest_finish_offset_min=60,
                    exclusive_group="fuel-safe",
                ),
            ),
            preferred_stand="S1",
        )
        coordinator.commit(coordinator.plan(occupied))

        plan_request = StandServiceRequest(
            StandDemand("F1", Window(D(10), D(12)), 40, False, buffer_before_min=0, buffer_after_min=0),
            (
                StandServiceSpec(
                    "fuel",
                    20,
                    preferred_offset_min=10,
                    latest_finish_offset_min=60,
                    exclusive_group="fuel-safe",
                ),
            ),
            preferred_stand="S3",
        )
        plan = coordinator.plan(plan_request)
        committed = coordinator.commit(plan)

        self.assertEqual(committed.assignment.stand_id, "S3")
        self.assertEqual(committed.services["fuel"].window.start, D(10, 30))

    def test_failed_commit_leaves_no_half_written_package(self):
        coordinator = make_coordinator()
        good = coordinator.plan(request(preferred_stand=None))
        tampered_assignment = good.assignment.__class__(
            good.assignment.flight_id,
            "MISSING",
            good.assignment.window,
            good.assignment.protected_window,
            good.assignment.generation,
            good.assignment.score,
        )
        bad = good.__class__(
            good.request,
            tampered_assignment,
            good.services,
            good.strategy,
            good.adjustment_min,
            good.generation,
        )

        with self.assertRaises(Exception):
            coordinator.commit(bad)

        self.assertIsNone(coordinator.package_for("F1"))
        self.assertIsNone(coordinator.stand_for("F1"))
        self.assertEqual(coordinator.services_for("F1"), ())
        self.assertEqual(coordinator.generation, 0)

        recovered = coordinator.commit(good)
        self.assertEqual(recovered.assignment.stand_id, "S1")
        self.assertEqual(len(coordinator.services_for("F1")), 2)

    def test_stale_plan_is_rejected(self):
        coordinator = make_coordinator()
        plan = coordinator.plan(request(preferred_stand=None))
        coordinator.release(coordinator.commit(plan).request.demand.flight_id)

        with self.assertRaises(StaleServicePlan):
            coordinator.commit(plan)

    def test_workflow_event_waits_for_successful_atomic_commit(self):
        coordinator = make_coordinator()
        control = RampWorkflowControl()
        calls = []

        def fail_commit():
            calls.append("commit")
            raise RuntimeError("joint constraint rejected")

        with self.assertRaises(RuntimeError):
            control.commit_stand_service_package(
                "op-1",
                "F1",
                D(9),
                fail_commit,
            )

        self.assertEqual(calls, ["commit"])
        self.assertEqual(control.stream.head, 0)
        self.assertEqual(control.receipts, {})


if __name__ == "__main__":
    unittest.main()
