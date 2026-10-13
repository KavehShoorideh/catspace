"""The frame's mate (vocab.rs frame_mate): a forced mate searched inside the frame -- our moves a check or a safe move
not raising the frame's measure, every reply of theirs -- asked when the frame's goal is mate; explained move by move."""
import catspace.incubator.analyst  # noqa: F401
import analyst_core as A

FRAME = ["(count (and (nearer L W) (not (between W L))))"]


def test_a_mate_in_two_inside_the_frame_with_its_words():
    r = A.frame_mate_of("3B4/8/8/8/8/7K/8/5N1k w - - 0 1", FRAME, True)
    assert r is not None
    move, words = r
    assert move == "f1g3" and "mate in 2" in words and words.rstrip().endswith("#") and "flight square" in words


def test_no_mate_no_claim():
    assert A.frame_mate_of("8/8/3k4/8/8/2K5/3BN3/8 w - - 0 1", FRAME, True) is None   # far from mate: nothing proved
