#!/usr/bin/python3
from unittest.mock import patch
from unittest.mock import MagicMock
from fractions import Fraction as F
import unittest
import sys
import datetime
import random
import rlab_common as rlcom
import rlab_common_numpy_pandas as rlnp
pd = rlnp.pd


def results_variance(result1, result2):
	ok = True
	c = 0
	v = 0
	for q, w in zip(result1, result2):
		 c += 1
		 if q != w:
			  # print(c, q, w)
			  v += abs(q - w)
			  ok = False
	v = float(v / c)
	print('+/- variance = %s' % v)
	return v


class TestStaticFixtures_Web():
	"""Below are static fixtures.
	"""
	def __init__(self):
		self.CMD = 'echo > /dev/null'
		self.WEB_PUT_OK = 'PUT-OK'
		self.WEB_POST_OK = 'POST-OK'
		self.WEB_GET_OK = 'GET-OK'

	def put(*args, **kwargs):
		self = args[0]
		request = args[1]
		url = request['url']
		data = request['data']
		if url == "https://api.endpoint.com:443/v1/change/order":
			return self.WEB_PUT_OK
		print('put() - CMD - %s' % cmd)
		raise RuntimeError('No test case for Web.put()')

	def post(*args, **kwargs):
		self = args[0]
		request = args[1]
		url = request['url']
		data = request['data']
		if url == "https://api.endpoint.com:443/v1/create/order":
			return self.WEB_POST_OK
		print('post() - CMD - %s' % cmd)
		raise RuntimeError('No test case for Web.post()')

	def get(*args, **kwargs):
		self = args[0]
		request = args[1]
		url = request['url']
		data = request['data']
		if url == "https://api.endpoint.com:443/v1/price/get":
			return self.WEB_GET_OK
		print('get() - CMD - %s' % cmd)
		raise RuntimeError('No test case for Web.get()')


WEB = rlcom.Web(**{
	'command': '/usr/bin/curl',
	'type': 'api-json',
	'headers': '-H "Authorization: Bearer 1234567890"',
	'url': 'https://api.endpoint.com:443'
})




FIXTURE_WEB = TestStaticFixtures_Web()
WEB.CMD = FIXTURE_WEB.CMD
WEB.put = FIXTURE_WEB.put
WEB.post = FIXTURE_WEB.post
WEB.get = FIXTURE_WEB.get

print('Finished setting up TestCase class...OK\n')


class TestRLabCommon_Web(unittest.TestCase):
	""" Test rlab_common module. Web() class.
	"""
	def shortDescription(self):
		"""Tests rlab_common module. Web() class.
		"""
		pass

	def setUp(self):
		pass

	def tearDown(self):
		pass

	@classmethod
	def setUpClass(cls):
		print('*** Starting TestRLabCommon_Web()...')

	@classmethod
	def tearDownClass(cls):
		print('*** Starting TestRLabCommon_Web()...DONE')

	def test_0__init__(self):
		# The first test to execute. Use as a constructor.
		print('Running non destructive static fixture tests...')

	def test_1_1_put_fail(self):
		result, expected = False, False
		# WEB.put = rlcom.Web().put
		# result = WEB.put(WEB.request(url='v1/change/order', units=101))
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_1_2_put(self):
		result, expected = None, FIXTURE_WEB.WEB_PUT_OK
		# WEB.put = FIXTURE_WEB.put
		# result = WEB.put(WEB.request(url='v1/change/order', units=101))
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)
	
	def test_2_1_post_fail(self):
		result, expected = False, False
		# WEB.post = rlcom.Web().post
		# result = WEB.post(WEB.request(url='v1/create/order', units=101))
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_2_2_post(self):
		result, expected = None, FIXTURE_WEB.WEB_POST_OK
		# WEB.post = FIXTURE_WEB.post
		# result = WEB.post(WEB.request(url='v1/create/order', units=101))
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_3_1_get_fail(self):
		result, expected = False, False
		# WEB.get = rlcom.Web().get
		# result = WEB.get(WEB.request(url='v1/price/get', units=101))
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_3_2_get(self):
		result, expected = None, FIXTURE_WEB.WEB_GET_OK
		# WEB.get = FIXTURE_WEB.get
		# result = WEB.get(WEB.request(url='v1/price/get', units=101))
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)


class TestStaticFixtures_VectorizationEngine():
	"""Below are static fixtures.
	"""
	def __init__(self):
		self._DATA_A1 = [round(random.randint(1, 1000) / 1000, 6) for x in range(0, 1000)]  # nosec
		self._DATA_A1_DF = pd.DataFrame(self._DATA_A1)
		self._DATA_A2 = [
			 self._DATA_A1[6:],
			 [round(float(x), 6) for x in self._DATA_A1_DF.shift(1).values][6:],
			 [round(float(x), 6) for x in self._DATA_A1_DF.shift(2).values][6:],
			 [round(float(x), 6) for x in self._DATA_A1_DF.shift(3).values][6:],
			 [round(float(x), 6) for x in self._DATA_A1_DF.shift(4).values][6:],
			 [round(float(x), 6) for x in self._DATA_A1_DF.shift(5).values][6:],
			 [round(float(x), 6) for x in self._DATA_A1_DF.shift(6).values][6:]
		]

		self._DATA_B1 = [round(random.randint(1, 1000) / 1000, 6) for x in range(0, 1000)]  # nosec
		self._DATA_B1_DF = pd.DataFrame(self._DATA_B1)
		self._DATA_B2 = [
			 self._DATA_B1[6:],
			 [round(float(x), 6) for x in self._DATA_B1_DF.shift(1).values][6:],
			 [round(float(x), 6) for x in self._DATA_B1_DF.shift(2).values][6:],
			 [round(float(x), 6) for x in self._DATA_B1_DF.shift(3).values][6:],
			 [round(float(x), 6) for x in self._DATA_B1_DF.shift(4).values][6:],
			 [round(float(x), 6) for x in self._DATA_B1_DF.shift(5).values][6:],
			 [round(float(x), 6) for x in self._DATA_B1_DF.shift(6).values][6:]
		]

		self._DATA_C1 = [
			[random.randint(1, 100) / 100 for x in range(0, 1000)],  # nosec
			[random.randint(1, 100) / 100 for x in range(0, 1000)],  # nosec
			[random.randint(1, 100) / 100 for x in range(0, 1000)]   # nosec
		]

		self._DATA_C2 = [
			[1 for x in range(0, 1000)],
			[2 for x in range(0, 1000)],
			[3 for x in range(0, 1000)]
		]


FIXTURE_VE = TestStaticFixtures_VectorizationEngine()
FIXTURE_VE_CEO = {'skip_post_cleaning': True, 'skip_pre_preparations': True}


class TestRLabCommon_VectorizationEngine(unittest.TestCase):
	""" Test VRollingSum
	"""
	def shortDescription(self):
		"""Tests VRollingSum
		"""
		pass

	def setUp(self):
		pass

	def tearDown(self):
		pass

	@classmethod
	def setUpClass(cls):
		print('*** Starting TestRLabCommon_VectorizationEngine()...')

	@classmethod
	def tearDownClass(cls):
		print('*** Starting TestRLabCommon_VectorizationEngine()...DONE')

	def test_0__init__(self):
		# The first test to execute. Use as a constructor.
		print('Running non destructive static fixture tests...')

	def test_1_avg_result_size_check1(self):
		data = FIXTURE_VE._DATA_A1
		result, expected = None, data.__len__()
		result = rlcom.avg3(data).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_1_avg_result_size_check2(self):
		data = FIXTURE_VE._DATA_A1
		result, expected = None, data.__len__()
		result = rlcom.avg64(data).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_1_avg_result_size_check3(self):
		data = FIXTURE_VE._DATA_A2
		result, expected = None, data[0].__len__()
		result = rlcom.avg7(data, **FIXTURE_VE_CEO).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_1_avg_pure_python_component_check(self):
		data = FIXTURE_VE._DATA_A1
		result_A = rlcom.avg7(data)
		data = FIXTURE_VE._DATA_A2
		result_B = rlcom.VectorizeEngine()._square_right_by_ref(
			result_A,
			[round(x, 6) for x in rlcom.avg7(data, **FIXTURE_VE_CEO)]
		)
		v = results_variance(result_A, result_B)
		threshold = 0.01
		self.assertTrue(
			v < threshold,
			'Expected both values to be under a variance of %s, \nRecieved - \n%s \n%s' % (
				threshold, result_A[:10], result_B[:10]
			)
		)

	def test_2_rtn_result_size_check1(self):
		data = FIXTURE_VE._DATA_A1
		result, expected = None, data.__len__()
		result = rlcom.rtn3(data, 'M', 'MON').__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_2_rtn_result_size_check2(self):
		data = FIXTURE_VE._DATA_A1
		result, expected = None, data.__len__()
		result = rlcom.rtn64(data, 'M', 'MON').__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_2_rtn_result_size_check3(self):
		data = FIXTURE_VE._DATA_A2
		result, expected = None, data[0].__len__()
		result = rlcom.rtn7(data, 'M', 'MON', **FIXTURE_VE_CEO).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_2_rtn_pure_python_component_check(self):
		data = FIXTURE_VE._DATA_A1
		result_A = rlcom.rtn7(data, 'M', 'MON')
		data = FIXTURE_VE._DATA_A2
		result_B = rlcom.VectorizeEngine()._square_right_by_ref(
			result_A,
			[round(x, 6) for x in rlcom.rtn7(data, 'M', 'MON', **FIXTURE_VE_CEO)]
		)
		v = results_variance(result_A, result_B)
		threshold = 0.01
		self.assertTrue(
			v < threshold,
			'Expected both values to be under a variance of %s, \nRecieved - \n%s \n%s' % (
				threshold, result_A[:10], result_B[:10]
			)
		)

	def test_3_min_result_size_check1(self):
		data = FIXTURE_VE._DATA_A1
		result, expected = None, data.__len__()
		result = rlcom.min3(data).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_3_min_result_size_check2(self):
		data = FIXTURE_VE._DATA_A1
		result, expected = None, data.__len__()
		result = rlcom.min64(data).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_3_min_result_size_check3(self):
		data = FIXTURE_VE._DATA_A2
		result, expected = None, data[0].__len__()
		result = rlcom.min7(data, **FIXTURE_VE_CEO).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_3_min_pure_python_component_check(self):
		data = FIXTURE_VE._DATA_A1
		result_A = rlcom.min7(data)
		data = FIXTURE_VE._DATA_A2
		result_B = rlcom.VectorizeEngine()._square_right_by_ref(
			result_A,
			[round(x, 6) for x in rlcom.min7(data, **FIXTURE_VE_CEO)]
		)
		v = results_variance(result_A, result_B)
		threshold = 0.01
		self.assertTrue(
			v < threshold,
			'Expected both values to be under a variance of %s, \nRecieved - \n%s \n%s' % (
				threshold, result_A[:10], result_B[:10]
			)
		)

	def test_4_max_result_size_check1(self):
		data = FIXTURE_VE._DATA_A1
		result, expected = None, data.__len__()
		result = rlcom.max3(data).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_4_max_result_size_check2(self):
		data = FIXTURE_VE._DATA_A1
		result, expected = None, data.__len__()
		result = rlcom.max64(data).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_4_max_result_size_check3(self):
		data = FIXTURE_VE._DATA_A2
		result, expected = None, data[0].__len__()
		result = rlcom.max7(data, **FIXTURE_VE_CEO).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_4_max_pure_python_component_check(self):
		data = FIXTURE_VE._DATA_A1
		result_A = rlcom.max7(data)
		data = FIXTURE_VE._DATA_A2
		result_B = rlcom.VectorizeEngine()._square_right_by_ref(
			result_A,
			[round(x, 6) for x in rlcom.max7(data, **FIXTURE_VE_CEO)]
		)
		v = results_variance(result_A, result_B)
		threshold = 0.01
		self.assertTrue(
			v < threshold,
			'Expected both values to be under a variance of %s, \nRecieved - \n%s \n%s' % (
				threshold, result_A[:10], result_B[:10]
			)
		)

	def test_5_mode_result_size_check1(self):
		import torch as pt
		data = FIXTURE_VE._DATA_A1
		result, expected = None, data.__len__()
		result = rlcom.mode3(data, use_pytorch_lib=pt).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_5_mode_result_size_check2(self):
		import numpy as np
		data = FIXTURE_VE._DATA_A1
		result, expected = None, data.__len__()
		result = rlcom.mode64(data, use_numpy_lib=np).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_5_mode_result_size_check3(self):
		import numpy as np
		data = FIXTURE_VE._DATA_A2
		result, expected = None, data[0].__len__()
		result = rlcom.mode7(data, use_numpy_lib=np, **FIXTURE_VE_CEO).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_5_mode_pure_python_component_check(self):
		import numpy as np
		data = FIXTURE_VE._DATA_A1
		result_A = rlcom.mode7(data, use_numpy_lib=np)
		data = FIXTURE_VE._DATA_A2
		result_B = rlcom.VectorizeEngine()._square_right_by_ref(
			result_A,
			[round(x, 6) for x in rlcom.mode7(data, use_numpy_lib=np, **FIXTURE_VE_CEO)]
		)
		v = results_variance(result_A, result_B)
		threshold = 0.01
		self.assertTrue(
			v < threshold,
			'Expected both values to be under a variance of %s, \nRecieved - \n%s \n%s' % (
				threshold, result_A[:10], result_B[:10]
			)
		)

	def test_6_cumsum_result_size_check1(self):
		data = FIXTURE_VE._DATA_A1
		result, expected = None, data.__len__()
		result = rlcom.cumsum3(data).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_6_cumsum_result_size_check2(self):
		data = FIXTURE_VE._DATA_A1
		result, expected = None, data.__len__()
		result = rlcom.cumsum64(data).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_6_cumsum_result_size_check3(self):
		data = FIXTURE_VE._DATA_A2
		result, expected = None, data[0].__len__()
		result = rlcom.cumsum7(data, **FIXTURE_VE_CEO).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_6_cumsum_pure_python_component_check(self):
		data = FIXTURE_VE._DATA_A1
		result_A = rlcom.cumsum7(data)
		data = FIXTURE_VE._DATA_A2
		result_B = rlcom.VectorizeEngine()._square_right_by_ref(
			result_A,
			[round(x, 6) for x in rlcom.cumsum7(data, **FIXTURE_VE_CEO)]
		)
		v = results_variance(result_A, result_B)
		threshold = 0.01
		self.assertTrue(
			v < threshold,
			'Expected both values to be under a variance of %s, \nRecieved - \n%s \n%s' % (
				threshold, result_A[:10], result_B[:10]
			)
		)

	def test_7_gr_result_size_check1(self):
		data = FIXTURE_VE._DATA_A1
		result, expected = None, data.__len__()
		result = rlcom.gr(data).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_7_gr_pandas_compare_check(self):
		data = FIXTURE_VE._DATA_A1
		data_df = FIXTURE_VE._DATA_A1_DF[0]
		result_A = rlcom.gr(data, dropna=True)
		result_PD = [round(x, 6) for x in data_df.pct_change().dropna().values]
		v = results_variance(result_A, result_PD)
		threshold = 0.01
		self.assertTrue(
			v < threshold,
			'Expected both values to be under a variance of %s, \nRecieved - \n%s \n%s' % (
				threshold, result_A[:10], result_PD[:10]
			)
		)

	def test_7_gr_accel_check(self):
		import torch as pt
		data = FIXTURE_VE._DATA_A1
		data_df = FIXTURE_VE._DATA_A1_DF[0]
		result_A = rlcom.gr(data, dropna=True, use_pytorch_lib=pt)
		result_PD = [round(x, 6) for x in data_df.pct_change().dropna().values]
		v = results_variance(result_A, result_PD)
		threshold = 0.01
		self.assertTrue(
			v < threshold,
			'Expected both values to be under a variance of %s, \nRecieved - \n%s \n%s' % (
				threshold, result_A[:10], result_PD[:10]
			)
		)

	def test_7_gr_pure_python_component_check(self):
		data = FIXTURE_VE._DATA_A1[5:]
		result_A = rlcom.gr(data, dropna=True)
		data = FIXTURE_VE._DATA_A2[:2].copy()
		result_B = rlcom.gr(data, **FIXTURE_VE_CEO)
		result_B = [round(x, 6) for x in result_B]
		v = results_variance(result_A, result_B)
		threshold = 0.01
		self.assertTrue(
			v < threshold,
			'Expected both values to be under a variance of %s, \nRecieved - \n%s \n%s' % (
				threshold, result_A[:10], result_B[:10]
			)
		)

	def test_8_var_result_size_check1(self):
		data = FIXTURE_VE._DATA_A1
		result, expected = None, data.__len__()
		result = rlcom.var3(data).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_8_var_result_size_check2(self):
		data = FIXTURE_VE._DATA_A1
		result, expected = None, data.__len__()
		result = rlcom.var64(data).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_8_var_result_size_check3(self):
		data = FIXTURE_VE._DATA_A2
		result, expected = None, data[0].__len__()
		result = rlcom.var7(data, **FIXTURE_VE_CEO).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_8_var_pure_python_component_check(self):
		data = FIXTURE_VE._DATA_A1
		result_A = rlcom.var7(data)
		data = FIXTURE_VE._DATA_A2
		result_B = rlcom.VectorizeEngine()._square_right_by_ref(
			result_A,
			[round(x, 6) for x in rlcom.var7(data, **FIXTURE_VE_CEO)]
		)
		v = results_variance(result_A, result_B)
		threshold = 0.01
		self.assertTrue(
			v < threshold,
			'Expected both values to be under a variance of %s, \nRecieved - \n%s \n%s' % (
				threshold, result_A[:10], result_B[:10]
			)
		)

	def test_8_var_accel_check(self):
		import torch as pt
		import numpy as np
		data = FIXTURE_VE._DATA_A1
		result_A = rlcom.var7(data, dropna=True)
		result_B = rlcom.var7(data, dropna=True, use_pytorch_lib=pt)
		result_C = rlcom.var7(data, dropna=True, use_numpy_lib=np)
		v1 = results_variance(result_A, result_B)
		v2 = results_variance(result_A, result_C)
		threshold = 0.01
		self.assertTrue(
			v1 < threshold and v2 < threshold,
			'Expected both values to be under a variance of ' +
			'%s, v1=%s, v2=%s, \nRecieved - \n%s \n%s' % (
				threshold, v1, v2, result_A[:10], result_B[:10]
			)
		)

	def test_9_std_result_size_check1(self):
		data = FIXTURE_VE._DATA_A1
		result, expected = None, data.__len__()
		result = rlcom.std3(data).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_9_std_result_size_check2(self):
		data = FIXTURE_VE._DATA_A1
		result, expected = None, data.__len__()
		result = rlcom.std64(data).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_9_std_result_size_check3(self):
		data = FIXTURE_VE._DATA_A2
		result, expected = None, data[0].__len__()
		result = rlcom.std7(data, **FIXTURE_VE_CEO).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_9_std_pure_python_component_check(self):
		data = FIXTURE_VE._DATA_A1
		result_A = rlcom.std7(data)
		data = FIXTURE_VE._DATA_A2
		result_B = rlcom.VectorizeEngine()._square_right_by_ref(
			result_A,
			[round(x, 6) for x in rlcom.std7(data, **FIXTURE_VE_CEO)]
		)
		v = results_variance(result_A, result_B)
		threshold = 0.01
		self.assertTrue(
			v < threshold,
			'Expected both values to be under a variance of %s, \nRecieved - \n%s \n%s' % (
				threshold, result_A[:10], result_B[:10]
			)
		)

	def test_9_std_pandas_compare_check(self):
		data = FIXTURE_VE._DATA_A1
		data_df = FIXTURE_VE._DATA_A1_DF[0]
		result_A = rlcom.std7(data, dropna=True)
		result_PD = [round(x, 6) for x in data_df.rolling(7).std().dropna().values]
		v = results_variance(result_A, result_PD)
		threshold = 0.01
		self.assertTrue(
			v < threshold,
			'Expected both values to be under a variance of %s, \nRecieved - \n%s \n%s' % (
				threshold, result_A[:10], result_PD[:10]
			)
		)

	def test_9_std_accel_check(self):
		import torch as pt
		import numpy as np
		data = FIXTURE_VE._DATA_A1
		result_A = rlcom.std7(data, dropna=True)
		result_B = rlcom.std7(data, dropna=True, use_pytorch_lib=pt)
		result_C = rlcom.std7(data, dropna=True, use_numpy_lib=np)
		v1 = results_variance(result_A, result_B)
		v2 = results_variance(result_A, result_C)
		threshold = 0.01
		self.assertTrue(
			v1 < threshold and v2 < threshold,
			'Expected both values to be under a variance of ' +
			'%s, v1=%s, v2=%s, \nRecieved - \n%s \n%s' % (
				threshold, v1, v2, result_A[:10], result_B[:10]
			)
		)

	def test_10_bme_result_size_check1(self):
		data = [FIXTURE_VE._DATA_A1, FIXTURE_VE._DATA_B1]
		result, expected = None, data[1].__len__()
		result = rlcom.bme3(data).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_10_bme_result_size_check2(self):
		data = [FIXTURE_VE._DATA_A1, FIXTURE_VE._DATA_B1]
		result, expected = None, data[1].__len__()
		result = rlcom.bme64(data).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_10_bme_result_size_check3(self):
		data = [FIXTURE_VE._DATA_A1, FIXTURE_VE._DATA_B1]
		ve = rlcom.VectorizeEngine()
		tag, gr, vshift = rlcom.tag, rlcom.gr, ve._vshift
		data = [
			vshift(tag(gr(data[0])), 0, 7),
			vshift(tag(gr(data[1])), 0, 7)
		]
		result, expected = None, data[1][0].__len__()
		result = rlcom.bme7(data, **FIXTURE_VE_CEO).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_10_bme_pure_python_component_check(self):
		data = [FIXTURE_VE._DATA_A1, FIXTURE_VE._DATA_B1]
		result_A = rlcom.bme7(data)
		data = [FIXTURE_VE._DATA_A1, FIXTURE_VE._DATA_B1]
		ve = rlcom.VectorizeEngine()
		tag, gr, vshift = rlcom.tag, rlcom.gr, ve._vshift
		data = [
			vshift(tag(gr(data[0])), 0, 7),
			vshift(tag(gr(data[1])), 0, 7)
		]
		rtn = [[round(float(y), 6) for y in x] for x in rlcom.bme7(data, **FIXTURE_VE_CEO) if x]
		nones = result_A.__len__() - rtn.__len__()
		fillnone = [None for x in range(0, nones)]
		fillnone.extend(rtn)
		result_B = tuple(fillnone)
		self.assertTrue(
			result_A == result_B,
			'Expected a value of %s\nRecieved a value of %s' % (result_B[:10], result_A[:10])
		)

	def test_11_bme_lockstep_result_size_check1(self):
		data = [FIXTURE_VE._DATA_A1, FIXTURE_VE._DATA_B1]
		result, expected = None, data[1].__len__()
		result = rlcom.bme_lockstep3(data, 2).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_11_bme_lockstep_result_size_check2(self):
		data = [FIXTURE_VE._DATA_A1, FIXTURE_VE._DATA_B1]
		result, expected = None, data[1].__len__()
		result = rlcom.bme_lockstep64(data, 9).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_11_bme_lockstep_result_size_check3(self):
		data = [FIXTURE_VE._DATA_A1, FIXTURE_VE._DATA_B1]
		get_rolling_data = rlcom.VLockStepBayesianMatrixEngine(
			data, 7, 3, **FIXTURE_VE_CEO
		).get_rolling_data
		s, b = get_rolling_data()
		result, expected = None, b[0].__len__()
		result = rlcom.bme_lockstep7([s, b], 3, **FIXTURE_VE_CEO).__len__()
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_11_bme_lockstep_pure_python_component_check(self):
		data = [FIXTURE_VE._DATA_A1, FIXTURE_VE._DATA_B1]
		result_A = rlcom.bme_lockstep7(data, 4)
		get_rolling_data = rlcom.VLockStepBayesianMatrixEngine(
			data, 7, 4, **FIXTURE_VE_CEO
		).get_rolling_data
		s, b = get_rolling_data()
		rtn = [
			[
				round(float(y), 6) for y in x
			] for x in rlcom.bme_lockstep7([s, b], 4, **FIXTURE_VE_CEO) if x
		]
		nones = result_A.__len__() - rtn.__len__()
		fillnone = [None for x in range(0, nones)]
		fillnone.extend(rtn)
		result_B = tuple(fillnone)
		self.assertTrue(
			result_A == result_B,
			'Expected a value of %s\nRecieved a value of %s' % (result_B[:10], result_A[:10])
		)


class TestRLabCommonNumpyPandas(unittest.TestCase):
	""" Test rlab_common_numpy_pandas module.
	"""
	def shortDescription(self):
		"""Tests rlab_common_numpy_pandas module.
		"""
		pass

	def setUp(self):
		pass

	def tearDown(self):
		pass

	@classmethod
	def setUpClass(cls):
		print('*** Starting TestRLabCommonNumpyPandas()...')

	@classmethod
	def tearDownClass(cls):
		print('*** Starting TestRLabCommonNumpyPandas()...DONE')

	def test_0__init__(self):
		# The first test to execute. Use as a constructor.
		print('Running non destructive static fixture tests...')

	def test_1_to_log(self):
		expected = -0.69314718
		x = F(1 / 2)
		result = rlnp.Normalise().to_log(x)
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_2_to_get_timeframe_by_date_monthly_detection(self):
		d1 = '2020-01-01'
		d2 = '2020-03-01'
		d3 = '2020-04-01'
		expected = 'monthly'
		result = rlnp.get_timeframe_by_dates(d1, d2, d3)
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_2_to_get_timeframe_by_date_quarterly_detection(self):
		d1 = '2019-08-01'
		d2 = '2020-01-01'
		d3 = '2020-04-01'
		expected = 'quarterly'
		result = rlnp.get_timeframe_by_dates(d1, d2, d3)
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_2_to_get_timeframe_by_date_daily_detection(self):
		d1 = '2020-01-01'
		d2 = '2020-01-03'
		d3 = '2020-01-04'
		expected = 'daily'
		result = rlnp.get_timeframe_by_dates(d1, d2, d3)
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_2_to_get_timeframe_by_date_weekly_detection(self):
		d1 = '2020-01-01'
		d2 = '2020-01-15'
		d3 = '2020-01-23'
		expected = 'weekly'
		result = rlnp.get_timeframe_by_dates(d1, d2, d3)
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_2_to_get_timeframe_by_date_yearly_detection(self):
		d1 = '2020-01-01'
		d2 = '2022-01-01'
		d3 = '2023-01-01'
		expected = 'yearly'
		result = rlnp.get_timeframe_by_dates(d1, d2, d3)
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_3_get_growth_rate_key_value(self):
		data = {
			'date': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
			'value': [-1, 0.1, 4, 8, 4, -4, -8, -4, -2, 4]
		}
		# expected = [0.1, 39.0, 1.0, -0.5, -2.0, 1.0, -0.5, -0.5, 2.0]  # Custom gr. BAD
		expected = [-1.1, 39.0, 1.0, -0.5, -2.0, 1.0, -0.5, -0.5, -3.0]  # Pandas gr GOOD
		df = pd.DataFrame(data)
		df_result = rlnp.get_growth_rate(df)
		arr_result = df_result[['gr']].values
		arr = []
		for v in arr_result:
			arr.append(round(v[0], 4))
		result = arr[1:]
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_3_get_growth_rate_ohlc(self):
		data = {
			'date': [1, 2, 3, 4, 5, 6],
			'open': [1, 9, 3, 8, 2, 11],
			'high': [5, 6, 7, 4, 5, 3],
			'low': [5, 6, 7, 4, 5, 3],
			'close': [5, 6, 7, 4, 5, 3]
		}
		# expected = [4.0, -0.3333, 1.3333, -0.5, 1.5, -0.7273]  # Custom gr BAD
		expected = [0.2, 0.1667, -0.4286, 0.25, -0.4]  # Custom gr BAD
		df = pd.DataFrame(data)
		df_result = rlnp.get_growth_rate(df)
		arr_result = df_result[['gr']].values
		arr = []
		for v in arr_result:
			arr.append(round(v[0], 4))
		result = arr[1:]
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_3_normalise_timeseries_check_returned_dates(self):
		data = {
			'date': [
				'2020-02-01', '2020-02-02', '2020-02-03', '2020-02-04', '2020-02-05',
				'2020-02-06', '2020-02-07', '2020-02-08', '2020-02-09', '2020-02-10',
				'2020-02-11', '2020-02-12'
			],
			'close': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14]
		}
		expected = [
			'2020-02-01', '2020-02-02', '2020-02-03', '2020-02-04', '2020-02-05', '2020-02-06',
			'2020-02-07', '2020-02-08', '2020-02-09', '2020-02-10', '2020-02-11', '2020-02-12'
		]
		df = pd.DataFrame(data)
		df_result = rlnp.normalise_timeseries_df(df)
		arr_result = df_result['date'].values
		arr = []
		for v in arr_result:
			arr.append(pd.to_datetime(v).strftime('%Y-%m-%d'))
		result = arr
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_3_normalise_timeseries_check_close(self):
		data = {
			'date': [
				'2020-02-01', '2020-02-02', '2020-02-03', '2020-02-04', '2020-02-05',
				'2020-02-06', '2020-02-07', '2020-02-08', '2020-02-09', '2020-02-10',
				'2020-02-11', '2020-02-12'
			],
			'close': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14]
		}
		df = pd.DataFrame(data)
		df_result = rlnp.normalise_timeseries_df(df)
		expected = [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14]
		arr_result = df_result['close'].values
		arr = []
		for v in arr_result:
			arr.append(v)
		result = arr
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_4_get_average_candle_body(self):
		data = {
			'date': [1, 2, 3, 4, 5, 6],
			'open': [5, 6, 7, 6, 5, 3],
			'high': [6, 8, 11, 12, 9, 5],
			'low': [2, 4, 2, 4, 4, 1],
			'close': [6, 7, 6, 5, 3, 4]
		}
		expected = [1.0, 1.0, 1.3333, 1.3333]
		df = pd.DataFrame(data)
		df_result = rlnp.get_average_candle_body(3, df)
		arr_result = df_result[['avg_body_size']].values
		arr = []
		for v in arr_result:
			arr.append(round(v[0], 4))
		result = arr[2:]
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_5_get_average_candle_tail(self):
		data = {
			'date': [1, 2, 3, 4, 5, 6],
			'open': [5, 6, 7, 6, 5, 3],
			'high': [6, 8, 11, 12, 9, 5],
			'low': [2, 4, 2, 4, 4, 1],
			'close': [6, 7, 6, 5, 3, 4]
		}
		expected = [3.6667, 2.6667, 2.0, 1.6667]
		df = pd.DataFrame(data)
		df_result = rlnp.get_average_candle_tail(3, df)
		arr_result = df_result[['avg_tail_size']].values
		arr = []
		for v in arr_result:
			arr.append(round(v[0], 4))
		result = arr[2:]
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_6_get_average_candle_head(self):
		data = {
			'date': [1, 2, 3, 4, 5, 6],
			'open': [5, 6, 7, 6, 5, 3],
			'high': [6, 8, 11, 12, 9, 5],
			'low': [2, 4, 2, 4, 4, 1],
			'close': [6, 7, 6, 5, 3, 4]
		}
		expected = [2.3333, 4.0, 4.6667, 4.6667]
		df = pd.DataFrame(data)
		df_result = rlnp.get_average_candle_head(3, df)
		arr_result = df_result[['avg_head_size']].values
		arr = []
		for v in arr_result:
			arr.append(round(v[0], 4))
		result = arr[2:]
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_7_get_drawdown_metrics_key_value(self):
		data = {
			'date': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
			'value': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14]
		}
		expected = [2.5]
		df = pd.DataFrame(data)
		df_result = rlnp.get_drawdown_metrics(2, df)
		arr_result = df_result['avg_duration'][['value']].values
		arr = []
		for v in arr_result:
			arr.append(round(v[0], 4))
		result = arr[1:]
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_7_get_drawdown_metrics_ohlc(self):
		data = {
			'date': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
			'open': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14],
			'high': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14],
			'low': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14],
			'close': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14]
		}
		expected = [2.5]
		df = pd.DataFrame(data)
		df_result = rlnp.get_drawdown_metrics(2, df)
		arr_result = df_result['avg_duration'][['value']].values
		arr = []
		for v in arr_result:
			arr.append(round(v[0], 4))
		result = arr[1:]
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_8_get_runup_metrics_key_value(self):
		data = {
			'date': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
			'value': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14]
		}
		expected = [4.0]
		df = pd.DataFrame(data)
		df_result = rlnp.get_runup_metrics(1, df)
		arr_result = df_result['avg_duration'][['value']].values
		arr = []
		for v in arr_result:
			arr.append(round(v[0], 4))
		result = arr
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_9_get_runup_metrics_ohlc(self):
		data = {
			'date': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
			'open': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14],
			'high': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14],
			'low': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14],
			'close': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14]
		}
		expected = [4.0]
		df = pd.DataFrame(data)
		df_result = rlnp.get_runup_metrics(1, df)
		arr_result = df_result['avg_duration'][['value']].values
		arr = []
		for v in arr_result:
			arr.append(round(v[0], 4))
		result = arr
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_10_add_quarterofyear_df(self):
		data = {
			'date': [
				'2020-02-01', '2020-02-07', '2020-02-14', '2020-02-21', '2020-02-28',
				'2020-03-01', '2020-03-07', '2020-03-14', '2020-03-21', '2020-03-28',
				'2020-04-01', '2020-04-07'
			],
			'close': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14]
		}
		expected = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]
		df = pd.DataFrame(data)
		df_result = rlnp.add_quarterofyear_df(df)
		arr_result = df_result[['quarter_of_year']].values
		arr = []
		for v in arr_result:
			arr.append(round(v[0], 4))
		result = arr
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_11_add_monthofyear_df(self):
		data = {
			'date': [
				'2020-02-01', '2020-02-07', '2020-02-14', '2020-02-21', '2020-02-28',
				'2020-03-01', '2020-03-07', '2020-03-14', '2020-03-21', '2020-03-28',
				'2020-04-01', '2020-04-07'
			],
			'close': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14]
		}
		expected = [2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4]
		df = pd.DataFrame(data)
		df_result = rlnp.add_monthofyear_df(df)
		arr_result = df_result[['month_of_year']].values
		arr = []
		for v in arr_result:
			arr.append(round(v[0], 4))
		result = arr
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_12_add_dayofweek_df(self):
		data = {
			'date': [
				'2020-02-01', '2020-02-02', '2020-02-03', '2020-02-04', '2020-02-05',
				'2020-02-06', '2020-02-07', '2020-02-08', '2020-03-21', '2020-03-28',
				'2020-04-01', '2020-04-07'
			],
			'close': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14]
		}
		# Monday 0, Sunday 6
		expected = [5, 6, 0, 1, 2, 3, 4, 5, 5, 5, 2, 1]
		df = pd.DataFrame(data)
		df_result = rlnp.add_dayofweek_df(df)
		arr_result = df_result[['day_of_week']].values
		arr = []
		for v in arr_result:
			arr.append(round(v[0], 4))
		result = arr
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_13_add_weekofmonth_df(self):
		data = {
			'date': [
				'2020-02-01', '2020-02-07', '2020-02-14', '2020-02-21', '2020-02-28',
				'2020-03-01', '2020-03-07', '2020-03-14', '2020-03-21', '2020-03-28',
				'2020-04-01', '2020-04-07'
			],
			'close': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14]
		}
		expected = [1, 1, 2, 3, 4, 1, 1, 2, 3, 4, 1, 1]
		df = pd.DataFrame(data)
		df_result = rlnp.add_weekofmonth_df(df)
		arr_result = df_result[['week_of_month']].values
		arr = []
		for v in arr_result:
			arr.append(round(v[0], 4))
		result = arr
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_14_seasonality_row(self):
		data = {
			'date': [
				'2020-02-01', '2020-02-07', '2020-02-14', '2020-02-21', '2020-02-28',
				'2020-03-01', '2020-03-07', '2020-03-14', '2020-03-21', '2020-03-28',
				'2020-04-01', '2020-04-07'
			],
			'open': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14],
			'high': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14],
			'low': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14],
			'close': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14]
		}
		expected = [1, 1, 2, 3, 4, 1, 1, 2, 3, 4, 1, 1]
		df = pd.DataFrame(data)
		df_result = rlnp.seasonality_row(df, 'weekly')
		arr_result = df_result[['week_of_month']].values
		arr = []
		for v in arr_result:
			arr.append(round(v[0], 4))
		result = arr
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_15_get_seasonality_case1(self):
		data = {
			'date': [
				'2020-02-01', '2020-02-07', '2020-02-14', '2020-02-21', '2020-02-28',
				'2020-03-01', '2020-03-07', '2020-03-14', '2020-03-21', '2020-03-28',
				'2020-04-01', '2020-04-07'
			],
			'open': [8, 22, 19, 7, 9, 22, 11, 21, 15, 18, 9, 10],
			'high': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14],
			'low': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14],
			'close': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14]
		}
		# ==== Period Average Growth Rate Test ====
		# expected = [0.0, 0.0, 0.0, 0.404, 0.4222]  # With custom GR. BAD
		expected = [0.0742, 0.0742, 0.0742, 0.0774, 0.0801]
		df = pd.DataFrame(data)
		df_result = rlnp.get_seasonality(2, df)
		arr_result = df_result[['week_1']].values
		arr = []
		for v in arr_result:
			arr.append(round(v[0], 4))
		result = arr[7:]
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_15_get_seasonality_case2(self):
		data = {
			'date': [
				'2020-02-01', '2020-02-07', '2020-02-14', '2020-02-21', '2020-02-28',
				'2020-03-01', '2020-03-07', '2020-03-14', '2020-03-21', '2020-03-28',
				'2020-04-01', '2020-04-07'
			],
			'open': [8, 22, 19, 7, 9, 22, 11, 21, 15, 18, 9, 10],
			'high': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14],
			'low': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14],
			'close': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14]
		}
		expected = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0]
		df = pd.DataFrame(data)
		# 50% growth rates are insignificant
		df_result = rlnp.get_seasonality(2, df, insig_pct=0.5, method='count')
		arr_result = df_result[['week_2_insigc']].values
		arr = []
		for v in arr_result:
			arr.append(v[0])
		result = arr[2:]
		self.assertTrue(
			result == expected,
			'Expected a value of %s\nRecieved a value of %s' % (expected, result)
		)

	def test_15_get_seasonality_case3(self):
		data = {
			'date': [
				'2020-02-01', '2020-02-07', '2020-02-14', '2020-02-21', '2020-02-28',
				'2020-03-01', '2020-03-07', '2020-03-14', '2020-03-21', '2020-03-28',
				'2020-04-01', '2020-04-07'
			],
			'open': [8, 22, 19, 7, 9, 22, 11, 21, 15, 18, 9, 10],
			'high': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14],
			'low': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14],
			'close': [15, 14, 13, 12, 13, 14, 15, 16, 13, 12, 13, 14]
		}
		expected2 = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
		df = pd.DataFrame(data)
		# 50% growth rates are insignificant
		df_result = rlnp.get_seasonality(2, df, insig_pct=0.5, method='count')
		arr_result2 = df_result[['week_4_insigc']].values
		arr2 = []
		for v in arr_result2:
			arr2.append(v[0])
		result = arr2[4:]
		self.assertTrue(
			result == expected2,
			'Expected a value of %s\nRecieved a value of %s' % (expected2, result)
		)


if __name__ == '__main__':
	 unittest.main()
