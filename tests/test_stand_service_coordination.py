import unittest
from datetime import datetime, timezone

from rampflow.timeutil import Window
from rampflow.stand_service_coordination import (
    CoordinationConflict,
    CoordinationRequest,
    CoordinationService,
    CoordinationStand,
    CoordinationStrategy,
    StandServiceCoordinator,
)

UTC = timezone.utc
D = lambda h, m=0: datetime(2026, 1, 1, h, m, tzinfo=UTC)


def svc(service, start, finish, duration, resource=None, group=None, hard=True, preds=frozenset()):
    return CoordinationService(
        service=service,
        duration_min=duration,
        resource_id=resource,
        exclusion_group=group,
        earliest_offset_min=start,
        latest_finish_offset_min=finish,
        hard=hard,
        predecessors=preds,
    )


def stands():
    return [
        CoordinationStand("S1", "T", adjacent=frozenset({"S2"})),
        CoordinationStand("S2", "T", adjacent=frozenset({"S1"})),
        CoordinationStand("S3", "T", adjacent=frozenset()),
    ]


class CoordinationTests(unittest.TestCase):
    def test_basic_package_commits(self):
        c = StandServiceCoordinator(stands())
        req = CoordinationRequest(
            "F1", D(10), D(12),
            (svc("catering", 0, 120, 20, resource="TRUCK-C1"),
             svc("fueling", 0, 120, 20, resource="TRUCK-F1", group="hazard")),
            preferred_stand="S1",
        )
        plan = c.coordinate(req)
        self.assertEqual(plan.stand_id, "S1")
        self.assertEqual(set(plan.bookings), {"catering", "fueling"})
        self.assertFalse(plan.stand_changed)

    def test_each_alone_fits_but_combined_group_conflict_shifts(self):
        # Catering and fueling each have a free truck and the stand itself is
        # free, but they share an exclusion group (cannot operate at the door
        # simultaneously): keeping the stand forces one service to shift.
        c = StandServiceCoordinator(stands())
        req = CoordinationRequest(
            "F1", D(10), D(12),
            (svc("catering", 0, 120, 20, resource="TRUCK-C1", group="door-exclusion"),
             svc("fueling", 0, 120, 20, resource="TRUCK-F1", group="door-exclusion")),
            preferred_stand="S1",
        )
        plan = c.coordinate(req)
        self.assertEqual(plan.stand_id, "S1")
        self.assertEqual(plan.strategy, CoordinationStrategy.KEEP_STAND_SHIFT_SERVICES)
        # both services use free trucks yet share the door-exclusion group, so
        # they must be serialised and one of them had to move off its earliest
        # slot rather than the package failing.
        cat = plan.booking("catering").window
        fuel = plan.booking("fueling").window
        self.assertFalse(cat.overlaps(fuel))
        self.assertTrue(
            plan.booking("catering").shifted or plan.booking("fueling").shifted,
            "at least one exclusion-group service should have shifted",
        )
        self.assertEqual(min(cat.start, fuel.start), D(10))
        self.assertEqual(max(cat.end, fuel.end), D(10, 40))

    def test_adjacent_stand_restriction_forces_reallocation(self):
        # G already occupies S2. S1 itself is free and every service fits, but
        # S1 declares S2 an adjacent restriction, so keeping S1 fails; AUTO
        # moves the whole package to the isolated stand S3.
        layout = [
            CoordinationStand("S1", "T", adjacent=frozenset({"S2"})),
            CoordinationStand("S2", "T"),
            CoordinationStand("S3", "T"),
        ]
        c = StandServiceCoordinator(layout)
        c.coordinate(CoordinationRequest(
            "G", D(10), D(11), (svc("park", 0, 60, 30),), preferred_stand="S2",
        ))
        req = CoordinationRequest(
            "F1", D(10), D(11),
            (svc("fueling", 0, 60, 20, resource="F1", group="hazard"),),
            preferred_stand="S1",
        )
        plan = c.coordinate(req)
        self.assertEqual(plan.stand_id, "S3")
        self.assertTrue(plan.stand_changed)
        self.assertEqual(plan.strategy, CoordinationStrategy.REALLOCATE_STAND)

    def test_combined_adjacency_and_group_constraints_decide_and_commit_atomically(self):
        # The embarrassing case from the brief:
        #   * S1 itself can park F1;
        #   * catering truck and fuel truck are each individually free;
        #   * but S2 (adjacent to S1) is occupied by G during the window, and
        #     an earlier catering/fuel exclusion on the *destination* options
        #     closes S3 until late.
        # The system must DECIDE: first try keeping S1 (fails on adjacency),
        # then re-plan the whole package, and must never leave a half-written
        # package on failure.
        layout = [
            CoordinationStand("S1", adjacent=frozenset({"S2"})),
            CoordinationStand("S2"),
            CoordinationStand("S3"),
        ]
        c = StandServiceCoordinator(layout)  # AUTO
        # Neighbour on S2 -> keeping S1 violates the adjacent-stand envelope.
        c.coordinate(CoordinationRequest(
            "G", D(10), D(11), (svc("park", 0, 60, 30),), preferred_stand="S2",
        ))
        req = CoordinationRequest(
            "F1", D(10), D(12),
            (svc("catering", 0, 120, 20, resource="CT", group="door"),
             svc("fueling", 0, 120, 20, resource="FT", group="door")),
            preferred_stand="S1",
        )
        plan = c.coordinate(req)
        # AUTO abandoned S1 (adjacency) and re-homed the whole package on S3.
        self.assertEqual(plan.stand_id, "S3")
        self.assertEqual(plan.strategy, CoordinationStrategy.REALLOCATE_STAND)
        # both services present, serialised by the group, neither overlapping.
        cat = plan.booking("catering").window
        fuel = plan.booking("fueling").window
        self.assertFalse(cat.overlaps(fuel))
        # and the rejected S1 attempt left no residue.
        self.assertEqual([h.flight_id for h in c.occupancy("S1")], [])
        self.assertEqual(c.plan_for("F1").stand_id, "S3")

    def test_keep_strategy_does_not_move_stand(self):
        layout = [
            CoordinationStand("S1", adjacent=frozenset({"S2"})),
            CoordinationStand("S2"),
            CoordinationStand("S3"),
        ]
        c = StandServiceCoordinator(
            layout, strategy=CoordinationStrategy.KEEP_STAND_SHIFT_SERVICES
        )
        c.coordinate(CoordinationRequest(
            "G", D(10), D(11), (svc("park", 0, 60, 30),), preferred_stand="S2",
        ))
        req = CoordinationRequest(
            "F1", D(10), D(11),
            (svc("fueling", 0, 60, 20, group="hazard"),),
            preferred_stand="S1",
        )
        with self.assertRaises(CoordinationConflict) as cm:
            c.coordinate(req)
        self.assertTrue(
            any("adjacent-stand" in v for v in cm.exception.violations),
            cm.exception.violations,
        )
        # nothing committed for F1
        self.assertNotIn("F1", c._by_flight)
        self.assertEqual([h.flight_id for h in c.occupancy("S1")], [])

    def test_no_half_commit_on_failure(self):
        # Every stand directly occupied across F1's window: the package fails
        # and committed state must be exactly as before the call (no stand row,
        # no resource booking, no flight entry left half-written).
        c = StandServiceCoordinator([
            CoordinationStand("S1"),
            CoordinationStand("S2"),
        ])
        c.coordinate(CoordinationRequest(
            "G1", D(10), D(12), (svc("a", 0, 120, 10, resource="RA"),),
            preferred_stand="S1",
        ))
        c.coordinate(CoordinationRequest(
            "G2", D(10), D(12), (svc("b", 0, 120, 10, resource="RB"),),
            preferred_stand="S2",
        ))
        before_stands = {
            sid: [h.flight_id for h in c.occupancy(sid)] for sid in ("S1", "S2")
        }
        before_res = {
            rid: [h.flight_id for h in holds] for rid, holds in c._resources.items()
        }
        req = CoordinationRequest(
            "F1", D(10, 30), D(11, 30),
            (svc("c", 0, 60, 10, resource="R1"),
             svc("d", 0, 60, 10, resource="R2", group="grp")),
            preferred_stand="S1",
        )
        with self.assertRaises(CoordinationConflict):
            c.coordinate(req)
        after_stands = {
            sid: [h.flight_id for h in c.occupancy(sid)] for sid in ("S1", "S2")
        }
        self.assertEqual(before_stands, after_stands)
        self.assertEqual(set(before_res), set(c._resources))
        for rid, flights in before_res.items():
            self.assertEqual(flights, [h.flight_id for h in c._resources[rid]])
        self.assertNotIn("F1", c._by_flight)

    def test_predecessors_respected(self):
        c = StandServiceCoordinator(stands())
        req = CoordinationRequest(
            "F1", D(10), D(12),
            (svc("clean", 0, 60, 20),
             svc("board", 0, 120, 20, preds=frozenset({"clean"}))),
            preferred_stand="S1",
        )
        plan = c.coordinate(req)
        self.assertLessEqual(
            plan.booking("clean").window.end,
            plan.booking("board").window.start,
        )

    def test_hard_window_violation_blocks_package(self):
        # G fuels on S1 (hazard group) until 11:00. F1's fueling must finish by
        # 10:40, so shifting on S1 cannot help; AUTO reallocates the package to
        # another stand whose group is free.
        c = StandServiceCoordinator(stands())
        c.coordinate(CoordinationRequest(
            "G", D(10), D(12),
            (svc("fueling", 0, 90, 60, resource="FG", group="hazard"),),
            preferred_stand="S1",
        ))
        req = CoordinationRequest(
            "F1", D(10), D(12),
            (svc("fueling", 0, 40, 20, resource="FF", group="hazard"),),
            preferred_stand="S1",
        )
        plan = c.coordinate(req)
        self.assertNotEqual(plan.stand_id, "S1")

    def test_replan_is_atomic_and_keeps_old_on_failure(self):
        # Zero-buffer stands make the geometry tractable: a stand frees exactly
        # when the blocking flight leaves, but its hazard-group service is the
        # thing that serialises F1.
        layout = [
            CoordinationStand(sid, protected_before_min=0, protected_after_min=0)
            for sid in ("S1", "S2", "S3", "S4")
        ]
        c = StandServiceCoordinator(layout)
        # Hazard blockers occupy every stand until 10:10, each with a distinct
        # truck and a hazard-group service running the full stay.
        for gid, sid, truck in (
            ("G1", "S1", "T1"),
            ("G2", "S2", "T2"),
            ("G3", "S3", "T3"),
            ("G4", "S4", "T4"),
        ):
            c.coordinate(CoordinationRequest(
                gid, D(9, 30), D(10, 10),
                (svc("occ", 0, 40, 40, resource=truck, group="hazard"),),
                preferred_stand=sid,
                strategy=CoordinationStrategy.KEEP_STAND_SHIFT_SERVICES,
            ))
        # F1 arrives exactly at 10:10: stands are free but the group frees at
        # 10:10 too, so 20-minute fueling lands 10:10-10:30.
        original = c.coordinate(CoordinationRequest(
            "F1", D(10, 10), D(12),
            (svc("fueling", 0, 110, 20, resource="FF", group="hazard"),),
            preferred_stand="S1",
        ))
        self.assertEqual(
            original.booking("fueling").window, Window(D(10, 10), D(10, 30))
        )
        # Replan pulls on-block earlier to 09:50 while the hard fueling finish
        # stays 35 minutes after on-block (10:25).  Every stand's hazard group
        # is busy until 10:10, so the earliest 20-minute slot ends 10:30 ->
        # infeasible on all four stands.
        req = CoordinationRequest(
            "F1", D(9, 50), D(12),
            (svc("fueling", 0, 35, 20, resource="FF", group="hazard"),),
            preferred_stand="S1",
        )
        with self.assertRaises(CoordinationConflict):
            c.coordinate_replan(req)
        # old package retained untouched
        self.assertEqual(c.plan_for("F1").stand_id, original.stand_id)
        self.assertEqual(
            c.plan_for("F1").booking("fueling").window,
            original.booking("fueling").window,
        )

    def test_shared_resource_conflict_is_adjacent_independent(self):
        # Same truck wanted by two flights on different (non-adjacent) stands;
        # individually each stand is free, but the shared resource serialises.
        c = StandServiceCoordinator(stands())
        c.coordinate(CoordinationRequest(
            "G", D(10), D(11),
            (svc("fueling", 0, 60, 30, resource="SHARED"),),
            preferred_stand="S1",
        ))
        req = CoordinationRequest(
            "F1", D(10), D(11),
            (svc("fueling", 0, 60, 30, resource="SHARED"),),
            preferred_stand="S3",
        )
        plan = c.coordinate(req)
        self.assertEqual(plan.booking("fueling").window.start, D(10, 30))


if __name__ == "__main__":
    unittest.main()
