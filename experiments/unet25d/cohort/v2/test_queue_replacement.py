"""CPU-only accounting guards; no API calls or GPU allocations."""
import unittest
from .queue_replacement import released_reservation,SCIENCE_SOURCE,EXPECTED_SCIENCE,ORIGINAL

class ReservationTests(unittest.TestCase):
    def test_only_unused_cancelled_first_session_is_released(self):
        r=dict(kernel=ORIGINAL,session_no=1,source_sha=SCIENCE_SOURCE,science_sha256=EXPECTED_SCIENCE,initial_quota={'refresh_at':'same'},reserved_seconds=43200,session_seconds=43200)
        q=dict(used_seconds=0,refresh_at='same')
        self.assertEqual(released_reservation(r,'CANCEL_ACKNOWLEDGED',q,[]),0)
        cases=[('QUEUED',q,[],r),('RUNNING',q,[],r),('CANCEL_ACKNOWLEDGED',dict(q,used_seconds=1),[],r),('CANCEL_ACKNOWLEDGED',q,['unexpected'],r),('CANCEL_ACKNOWLEDGED',q,[],dict(r,queue_replacement={})),('CANCEL_ACKNOWLEDGED',q,[],dict(r,reserved_seconds=0)),('CANCEL_ACKNOWLEDGED',dict(q,refresh_at='nextweek'),[],r)]
        for status,quota,files,record in cases:
            with self.subTest(status=status,quota=quota,files=files,record=record),self.assertRaises(PermissionError):released_reservation(record,status,quota,files)

if __name__=='__main__':unittest.main()
