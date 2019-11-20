import pytest
from numpy import array
from numpy.testing import (assert_allclose, assert_almost_equal,
                           assert_array_equal)

from touvlo.nn_clsf_vect import (init_params, linear_forward,
                                 linear_activation_forward,
                                 L_model_forward)


class TestNeuralNetwork:

    def test_init_params_1(self):
        parameters = init_params([3, 2, 1])

        assert parameters['W1'].shape == (2, 3)
        assert parameters['b1'].shape == (2, 1)
        assert parameters['W2'].shape == (1, 2)
        assert parameters['b2'].shape == (1, 1)

    def test_init_params_2(self):
        parameters = init_params([5, 4, 3, 1])

        assert parameters['W1'].shape == (4, 5)
        assert parameters['b1'].shape == (4, 1)
        assert parameters['W2'].shape == (3, 4)
        assert parameters['b2'].shape == (3, 1)
        assert parameters['W3'].shape == (1, 3)
        assert parameters['b3'].shape == (1, 1)

    def test_init_params_3(self):
        parameters = init_params([10, 7, 5, 2, 1])

        assert parameters['W1'].shape == (7, 10)
        assert parameters['b1'].shape == (7, 1)
        assert parameters['W2'].shape == (5, 7)
        assert parameters['b2'].shape == (5, 1)
        assert parameters['W3'].shape == (2, 5)
        assert parameters['b3'].shape == (2, 1)
        assert parameters['W4'].shape == (1, 2)
        assert parameters['b4'].shape == (1, 1)

    def test_linear_forward1(self):
        A = array([[1.62434536, -0.61175641], [-0.52817175, -1.07296862],
                   [0.86540763, -2.3015387]])
        W = array([[1.74481176, -0.7612069, 0.3190391]])
        b = array([[-0.24937038]])

        Z, linear_cache = linear_forward(A, W, b)

        assert_allclose(Z, array([[3.26295, -1.23430]]),
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(linear_cache[0], A)
        assert_allclose(linear_cache[1], W)
        assert_allclose(linear_cache[2], b)

    def test_linear_forward2(self):
        A = array([[-0.00172428, -0.00877858],
                   [0.00042214, 0.00582815],
                   [-0.01100619, 0.01144724]])
        W = array([[0.00901591, 0.00502494, 0.00900856]])
        b = array([[-0.00683728]])

        Z, linear_cache = linear_forward(A, W, b)

        assert_allclose(Z, array([[-0.00695, -0.006784]]),
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(linear_cache[0], A)
        assert_allclose(linear_cache[1], W)
        assert_allclose(linear_cache[2], b)

    def test_linear_forward3(self):
        A = array([[0.00838983, 0.00931102],
                   [0.00285587, 0.00885141],
                   [-0.00754398, 0.01252868]])
        W = array([[0.0051293, -0.00298093, 0.00488518]])
        b = array([[-0.00075572]])

        Z, linear_cache = linear_forward(A, W, b)

        assert_allclose(Z, array([[-0.00075, -0.00067]]),
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(linear_cache[0], A)
        assert_allclose(linear_cache[1], W)
        assert_allclose(linear_cache[2], b)

    def test_linear_activation_forward1(self):
        A_prev = array([[-0.41675785, -0.05626683],
                        [-2.1361961, 1.64027081],
                        [-1.79343559, -0.84174737]])
        W = array([[0.50288142, -1.24528809, -1.05795222]])
        b = array([[-0.90900761]])

        A, cache = linear_activation_forward(A_prev, W, b,
                                             activation="sigmoid")

        assert_allclose(A, array([[0.96890023, 0.11013289]]),
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(cache[0][0], A_prev)
        assert_allclose(cache[0][1], W)
        assert_allclose(cache[0][2], b)
        assert_allclose(cache[1], array([[3.43896131, -2.08938436]]),
                        rtol=0, atol=0.0001, equal_nan=False)

    def test_linear_activation_forward2(self):
        A_prev = array([[-0.41675785, -0.05626683],
                        [-2.1361961, 1.64027081],
                        [-1.79343559, -0.84174737]])
        W = array([[0.50288142, -1.24528809, -1.05795222]])
        b = array([[-0.90900761]])

        A, cache = linear_activation_forward(A_prev, W, b,
                                             activation="relu")

        assert_allclose(A, array([[3.43896131, 0]]),
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(cache[0][0], A_prev)
        assert_allclose(cache[0][1], W)
        assert_allclose(cache[0][2], b)
        assert_allclose(cache[1], array([[3.43896131, -2.08938436]]),
                        rtol=0, atol=0.0001, equal_nan=False)

    def test_linear_activation_forward3(self):
        A_prev = array([[-0.00172428, -0.00877858],
                        [0.00042214, 0.00582815],
                        [-0.01100619, 0.01144724]])
        W = array([[0.00901591, 0.00502494, 0.00900856]])
        b = array([[-0.00683728]])

        A, cache = linear_activation_forward(A_prev, W, b,
                                             activation="sigmoid")

        assert_allclose(A, array([[0.49826254, 0.498304]]),
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(cache[0][0], A_prev)
        assert_allclose(cache[0][1], W)
        assert_allclose(cache[0][2], b)
        assert_allclose(cache[1], array([[-0.00695, -0.006784]]),
                        rtol=0, atol=0.0001, equal_nan=False)

    def test_linear_activation_forward4(self):
        A_prev = array([[-0.00172428, -0.00877858],
                        [0.00042214, 0.00582815],
                        [-0.01100619, 0.01144724]])
        W = array([[0.00901591, 0.00502494, 0.00900856]])
        b = array([[-0.00683728]])

        A, cache = linear_activation_forward(A_prev, W, b,
                                             activation="relu")

        assert_allclose(A, array([[0, 0]]),
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(cache[0][0], A_prev)
        assert_allclose(cache[0][1], W)
        assert_allclose(cache[0][2], b)
        assert_allclose(cache[1], array([[-0.00695, -0.006784]]),
                        rtol=0, atol=0.0001, equal_nan=False)

    def test_linear_activation_forward5(self):
        A_prev = array([[0.00838983, 0.00931102],
                        [0.00285587, 0.00885141],
                        [-0.00754398, 0.01252868]])
        W = array([[0.0051293, -0.00298093, 0.00488518]])
        b = array([[-0.00075572]])

        A, cache = linear_activation_forward(A_prev, W, b,
                                             activation="sigmoid")

        assert_allclose(A, array([[0.4997951, 0.49980945]]),
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(cache[0][0], A_prev)
        assert_allclose(cache[0][1], W)
        assert_allclose(cache[0][2], b)
        assert_allclose(cache[1], array([[-0.00075, -0.00067]]),
                        rtol=0, atol=0.0001, equal_nan=False)

    def test_linear_activation_forward6(self):
        A_prev = array([[0.00838983, 0.00931102],
                        [0.00285587, 0.00885141],
                        [-0.00754398, 0.01252868]])
        W = array([[0.0051293, -0.00298093, 0.00488518]])
        b = array([[-0.00075572]])

        A, cache = linear_activation_forward(A_prev, W, b,
                                             activation="relu")

        assert_allclose(A, array([[0, 0]]),
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(cache[0][0], A_prev)
        assert_allclose(cache[0][1], W)
        assert_allclose(cache[0][2], b)
        assert_allclose(cache[1], array([[-0.00075, -0.00067]]),
                        rtol=0, atol=0.0001, equal_nan=False)

    def test_L_model_forward1(self):
        parameters = {}
        parameters['W1'] = array([[0.35480861, 1.81259031, -1.3564758,
                                   -0.46363197, 0.82465384],
                                  [-1.17643148, 1.56448966, 0.71270509,
                                   -0.1810066, 0.53419953],
                                  [-0.58661296, -1.48185327, 0.85724762,
                                   0.94309899, 0.11444143],
                                  [-0.02195668, -2.12714455, -0.83440747,
                                   -0.46550831, 0.23371059]])
        parameters['b1'] = array([[1.38503523],
                                  [-0.51962709],
                                  [-0.78015214],
                                  [0.95560959]])
        parameters['W2'] = array([[-0.12673638, -1.36861282, 1.21848065,
                                   -0.85750144],
                                  [-0.56147088, -1.0335199, 0.35877096,
                                   1.07368134],
                                  [-0.37550472, 0.39636757, -0.47144628,
                                   2.33660781]])
        parameters['b2'] = array([[1.50278553], [-0.59545972], [0.52834106]])
        parameters['W3'] = array([[0.9398248, 0.42628539, -0.75815703]])
        parameters['b3'] = array([[-0.16236698]])

        X = array([[-0.31178367, 0.72900392, 0.21782079, -0.8990918],
                   [-2.48678065, 0.91325152, 1.12706373, -1.51409323],
                   [1.63929108, -0.4298936, 2.63128056, 0.60182225],
                   [-0.33588161, 1.23773784, 0.11112817, 0.12915125],
                   [0.07612761, -0.15512816, 0.63422534, 0.810655]])

        Z1 = array([[-5.23825714, 3.18040136, 0.4074501, -1.88612721],
                    [-2.77358234, -0.56177316, 3.18141623, -0.99209432],
                    [4.18500916, -1.78006909, -0.14502619, 2.72141638],
                    [5.05850802, -1.25674082, -3.54566654, 3.82321852]])

        A1 = array([[0., 3.18040136, 0.4074501, 0.],
                    [0., 0., 3.18141623, 0.],
                    [4.18500916, 0., 0., 2.72141638],
                    [5.05850802, 0., 0., 3.82321852]])

        Z2 = array([[2.2644603, 1.09971298, -2.90298027, 1.54036335],
                    [6.33722569, -2.38116246, -4.11228806, 4.48582383],
                    [10.37508342, -0.66591468, 1.63635185, 8.17870169]])

        A2 = array([[2.2644603, 1.09971298, 0., 1.54036335],
                    [6.33722569, 0., 0., 4.48582383],
                    [10.37508342, 0., 1.63635185, 8.17870169]])

        Z3 = array([[-3.19864676, 0.87117055, -1.40297864, -3.00319435]])

        AL, caches = L_model_forward(X, parameters)

        assert_allclose(AL,
                        array([[0.03921668, 0.70498921, 
                                0.19734387, 0.04728177]]),
                        rtol=0, atol=0.0001, equal_nan=False)

        # caches = (cache1, cache2, cache3)
        assert len(caches) == 3

        # cache1 = ((X, W1, b1), Z1)
        assert_allclose(caches[0][0][0], X)
        assert_allclose(caches[0][0][1], parameters['W1'])
        assert_allclose(caches[0][0][2], parameters['b1'])
        assert_allclose(caches[0][1], Z1)

        # cache2 = ((A1, W2, b2), Z2)
        assert_allclose(caches[1][0][0], A1)
        assert_allclose(caches[1][0][1], parameters['W2'])
        assert_allclose(caches[1][0][2], parameters['b2'])
        assert_allclose(caches[1][1], Z2)

        # cache3 = ((A2, W3, b3), Z3)
        assert_allclose(caches[2][0][0], A2)
        assert_allclose(caches[2][0][1], parameters['W3'])
        assert_allclose(caches[2][0][2], parameters['b3'])
        assert_allclose(caches[2][1], Z3)

    def test_L_model_forward2(self):
        parameters = {}
        parameters['W1'] = array([[0.0072681, 0.00444083, -0.00856823,
                                   0.00446928, -0.01014648],
                                  [-0.02132323, 0.00173863, 0.00951201,
                                   0.00441897, 0.01469017],
                                  [0.01749516, 0.00353531, -0.00643337,
                                   -0.00047237, -0.0144904],
                                  [-0.0003619, -0.00090847, 0.0017629,
                                   0.0109462, -0.02126475]])
        parameters['b1'] = array([[0.00751449], [-0.00540607],
                                  [0.00793222], [0.00173653]])
        parameters['W2'] = array([[-0.01035434, 0.00874268, -0.00739572,
                                   0.00522945],
                                  [-0.00591876, -0.00477487, 0.0011253,
                                   0.01904742],
                                  [0.00694153, -0.00019581, 0.01662843,
                                   0.00030608]])
        parameters['b2'] = array([[-0.00297499], [-0.00968138],
                                  [0.00167067]])
        parameters['W3'] = array([[0.00116602, -0.00682257, -0.01914021]])
        parameters['b3'] = array([[-0.00139902]])

        X = array([[-0.31178367, 0.72900392, 0.21782079, -0.8990918],
                   [-2.48678065, 0.91325152, 1.12706373, -1.51409323],
                   [1.63929108, -0.4298936, 2.63128056, 0.60182225],
                   [-0.33588161, 1.23773784, 0.11112817, 0.12915125],
                   [0.07612761, -0.15512816, 0.63422534, 0.810655]])

        Z1 = array([[-0.02211435, 0.02765779, -0.01438117, -0.01854866],
                    [0.01214561, -0.02026147, 0.02674556, 0.02933695],
                    [-0.01780464, 0.02834375, -0.01044313, -0.02882979],
                    [0.00170298, 0.01673248, -0.00699771, -0.01132628]])

        A1 = array([[0., 0.02765779, 0., 0.],
                    [0.01214561, 0., 0.02674556, 0.02933695],
                    [0., 0.02834375, 0., 0.],
                    [0.00170298, 0.01673248, 0., 0.]])

        Z2 = array([[-0.0028599, -0.00338349, -0.00274116, -0.00271851],
                    [-0.00970693, -0.00949447, -0.00980908, -0.00982146],
                    [0.00166881, 0.00233909, 0.00166543, 0.00166492]])

        A2 = array([[0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [0.00166881, 0.00233909, 0.00166543, 0.00166492]])

        Z3 = array([[-0.00143096, -0.00144379, -0.0014309, -0.00143089]])

        AL, caches = L_model_forward(X, parameters)

        assert_allclose(AL,
                        array([[0.49964226, 0.49963905,
                                0.49964228, 0.49964228]]),
                        rtol=0, atol=0.0001, equal_nan=False)

        # caches = (cache1, cache2, cache3)
        assert len(caches) == 3

        # cache1 = ((X, W1, b1), Z1)
        assert_allclose(caches[0][0][0], X)
        assert_allclose(caches[0][0][1], parameters['W1'])
        assert_allclose(caches[0][0][2], parameters['b1'])
        assert_allclose(caches[0][1], Z1,
                        rtol=0, atol=0.0001, equal_nan=False)

        # cache2 = ((A1, W2, b2), Z2)
        assert_allclose(caches[1][0][0], A1,
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(caches[1][0][1], parameters['W2'])
        assert_allclose(caches[1][0][2], parameters['b2'])
        assert_allclose(caches[1][1], Z2,
                        rtol=0, atol=0.0001, equal_nan=False)

        # cache3 = ((A2, W3, b3), Z3)
        assert_allclose(caches[2][0][0], A2,
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(caches[2][0][1], parameters['W3'])
        assert_allclose(caches[2][0][2], parameters['b3'])
        assert_allclose(caches[2][1], Z3,
                        rtol=0, atol=0.0001, equal_nan=False)
