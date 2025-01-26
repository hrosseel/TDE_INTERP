import numpy as np
from lib.frac_delay import thiran


def test_thiran_allpass():

    # Test for delay = 0.8, order = 1
    delay = 0.8
    order = 1
    b, a = thiran(delay, order)
    expected_a = np.array([1, 1/9])
    expected_b = expected_a[::-1]
    assert np.allclose(b, expected_b)
    assert np.allclose(a, expected_a)

    # Test for delay = 3.6, order = 3
    delay = 2.6
    order = 3
    b, a = thiran(delay, order)
    expected_a = np.array([1, 1/3, -1/23, 2/483])
    expected_b = expected_a[::-1]
    assert np.allclose(b, expected_b)
    assert np.allclose(a, expected_a)

    # Test for delay = 16.9, order = 17
    delay = 16.9
    order = 17
    b, a = thiran(delay, order)
    expected_a = np.array([1, 17/179, -136/3759, 49/2837, -87/10372, 60/15449,
                           -88/52947, 151/234200, -20/89553, 9/132119,
                           -1/55462, 1/245617, -1/1302447, 1/8508592,
                           -1/71333663, 1/818540958, -1/14459059066,
                           1/524072688022])
    expected_b = expected_a[::-1]
    assert np.allclose(b, expected_b)
    assert np.allclose(a, expected_a)


test_thiran_allpass()
