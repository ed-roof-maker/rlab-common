"""
 ===============================================================================================
 RLab Common
 ===============================================================================================
 This module should limit use of thirdparty packages.
 Only use built in packages for this module.

"""

import os
import sys
import zipfile
import time
import traceback
import random
import subprocess  # nosec
# import datetime
import math
from fractions import Fraction as F

np = None
try:
	import numpy as np
except Exception:
	print('Numpy not found. Not required unless you need a vectorized MODE computation.')

try:
	import tensorflow as tf
	if tf.__name__ == 'tensorflow':
		print('Tensorflow found.')
except Exception:
	print('Tensorflow not found. Not required. Pure python functions used only.')

try:
	import torch as pt
	if pt.__name__ == 'torch':
		print('PyTorch Found')
except Exception:
	print('PyTorch not found. Not required. Pure python functions used only.')


DEBUG = False
VERBOSE = False
AT = '/opt/admin-tools'


# ===============================================================================================
# MODULE HELPER METHODS
# ===============================================================================================
def bytes_human(num):
	""" Convert bytes to readble unit of measures.
	"""
	for x in ['bytes', 'KB', 'MB', 'GB']:
		if num < 1024.0:
			return "%3.1f%s" % (num, x)
		num /= 1024.0
	return "%3.1f%s" % (num, 'TB')


def timeformat(a, b, c, d):
	""" INPUT: days, hours, minutes, seconds
		 OUTPUT: string - description formated in a pretty way
	"""
	return "%02d days, %02d hours, %02d minutes, %02d seconds" % (a, b, c, d)


def seconds_human(seconds):
	""" INPUT: seconds
		 OUTPUT: pretty printed string - days, hours, minutes, seconds
	"""
	d = 0
	h = 0
	m = 0
	s = 0
	one_minute = 60
	one_hour = 60 * 60
	one_day = 60 * 60 * 24
	if seconds < one_minute:
		return timeformat(0, 0, 0, seconds)
	if seconds >= one_minute and seconds < one_hour:
		s = seconds % one_minute  # get residule seconds
		m = seconds / one_minute  # get total minutes within an hour
		return timeformat(0, 0, m, s)
	if seconds >= one_hour and seconds < one_day:
		h = seconds / one_hour  # get total hours within a day
		m = (seconds / one_minute) % one_minute  # get residule minutes
		s = seconds % one_hour  # get residule seconds
		return timeformat(0, h, m, s)
	if seconds >= one_day:
		d = seconds / one_day  # get total days within a large set of seconds
		h = (seconds / one_hour) % one_hour  # get residule hours
		m = (seconds / one_hour / one_minute) % one_minute  # get residule minutes
		s = seconds % one_day  # get residule seconds
		return timeformat(d, h, m, s)
	raise ValueError('Bad calculation')


def rndgen_string(return_hash=False):
	""" Use the system's uuidgen to get a random string.
	"""
	# check if /opt/admin-tools/rndgen exists
	# use bash script tool to generate string uses /dev/random
	# system will use havged / rngd where necessary to seed the random entropy pool
	# tool = '/opt/admin-tools/rndgen'
	tool = '/usr/bin/uuidgen'
	if filesys().already_exists(tool):
		if return_hash:
			tool = tool + ' | sha512sum | cut -d " " -f 1'
		cmd = 'cat /proc/sys/kernel/random/entropy_avail'
		with subprocess.Popen(
				cmd, stdout=subprocess.PIPE, shell=False, encoding='utf8') as process:  # nosec
			entropy_pool = int(process.communicate()[0][:-1])
			if entropy_pool < 700:
				print(
					'WARNING system entropy pool is below 700, currently at ' + str(entropy_pool) +
					', there maybe not enough randomness. ' +
					'/dev/random will block if there is not enough entropy.'
				)
		with subprocess.Popen(
				tool, stdout=subprocess.PIPE, shell=False, encoding='utf8') as process:  # nosec
			return process.communicate()[0][:-1]
	print('ERROR custom random gen tool not found in ' + tool)
	sys.exit(1)


def rndgen_number(start, end):
	"""
		Randomly generate a number between start and end
		Uses custom bash rndgen script to leverage system entropy pool gatherers.
		Combines raw /dev/random and openssl random implementations to perform seeding.
		Reseeds per call.
		DO NOT USE FOR CRYPTOGRAPHIC PURPOSES!
	"""
	seed = time.time()
	seed = rndgen_string()
	random.seed(a=seed)
	return random.randint(start, end)  # nosec


def shuffle_list(var_list):
	"""
		Shuffle the items in var_list
		Uses custom bash rndgen script to leverage system entropy pool gatherers.
		Combines raw /dev/random and openssl random implementations to perform seeding.
		Reseeds per call.
		DO NOT USE FOR CRYPTOGRAPHIC PURPOSES!
	"""
	seed = time.time()
	nseed = rndgen_string()
	if nseed:
		seed = nseed
	random.seed(a=seed)
	random.shuffle(var_list)
	return var_list


def rndpick(var_list, quantity):
	"""
		Randomly pick X items from var_list
		DO NOT USE FOR CRYPTOGRAPHIC PURPOSES!
	"""
	random.seed(a=rndgen_string())
	return random.sample(var_list, quantity)


# ===============================================================================================
# MODULE CLASSES
# ===============================================================================================
class filesys():
	""" Tightly coupled to unix systems only. No windows.
	"""
	def whoami(self):
		""" Calls the system's whoami command.
		"""
		try:
			cmd = '/usr/bin/whoami'
			with subprocess.Popen(
					cmd, stdout=subprocess.PIPE, shell=False, encoding='utf8') as process:  # nosec
				rtn = process.communicate(timeout=9000)[0].split('\n', maxsplit=1)[0]
			return rtn

		except Exception as e:
			print('ERROR: Could not execute unix command ' + cmd)
			print(
				'ERROR: Exception for whoami() ERR - \n' + str(traceback.format_exc()) +
				'\n' + str(e)
			)
			return False

	def already_exists(self, path):
		""" Check is path exists. For pretty syntax only.
		"""
		if os.path.isdir(path):
			# print('Already Exists: %s' %(path))
			return True
		if os.path.isfile(path):
			# print('Already Exists: %s' %(path))
			return True
		# print('Does Not Exist: %s' %(path))
		return False

	def make_dir(self, path):
		""" Makes a directory. For pretty syntax only.
		"""
		os.makedirs(path)

	def is_utf(self, path):
		""" Check if file is utf.
		"""
		process = None
		if self.already_exists(path):
			cmd = '/usr/bin/file ' + path + ' | grep UTF | wc -l'
			with subprocess.Popen(
					cmd, stdout=subprocess.PIPE, shell=False, encoding='utf8') as process:  # nosec
				arr = process.communicate()[0].split('\n')
				items_arr = []
				for file_item in arr:
					if file_item == '':
						continue
					items_arr.append(file_item)
				match_count = int(items_arr[0])
				if match_count == 1:
					return True
				return False
		if VERBOSE:
			print('INFO: is_utf() - File path does not exist: %s' % (path))
		return False

	def chardetect(self, path):
		""" Run's the system command chardetect.
		"""
		process = None
		if self.already_exists(path):
			cmd = (
				'. ' + AT + '/set-env-py2;FN=$(/usr/bin/uuidgen);cat ' + path +
				' | tail -n 10 | tee /tmp/$FN;/usr/bin/chardetect /tmp/$FN | ' +
				'cut -d : -f 2 | cut -d " " -f 2;sleep 5;rm -f /tmp/$FN'
			)
			with subprocess.Popen(
					cmd, stdout=subprocess.PIPE, shell=False, encoding='utf8') as process:  # nosec
				arr = process.communicate()[0].split('\n')
				items_arr = []
				for file_item in arr:
					if file_item == '':
						continue
					items_arr.append(file_item)
				return str(items_arr[0])
		if VERBOSE:
			print('INFO: chardetect() - File path does not exist: %s' % (path))
		return False

	def save_utf(self, src_path, dst_path):
		""" Saves existing file as a utf file.
		"""
		with open(src_path, encoding='utf-8') as r:
			try:
				with open(dst_path, 'wb', encoding='utf-8') as w:
					w.write(b'\xEF\xBB\xBF' + r.read().encode('utf-8'))  # Saves as UTF8 by default
				if self.is_utf(dst_path):
					# print('Saved file as utf in: %s' % dst_path)
					pass
				else:
					if VERBOSE:
						print('ERROR Could not save file as UTF8 - ' + dst_path)
					return False
			except Exception:
				print('ERROR Could not open source path: ' + src_path)
				raise
		return True

	def remove(self, path):
		""" Removes the specified file.
		"""
		if self.already_exists(path):
			# os.remove(path)  # Does not do recursive deletes, but it is secure.
			cmd = '/usr/bin/rm -fr ' + path
			with subprocess.Popen(
					cmd, stdout=subprocess.PIPE, shell=False, encoding='utf8') as process:  # nosec
				process.communicate()
				return True
		if VERBOSE:
			print('Nothing to remove for %s' % (path))
		return False

	def ls_wildcard(self, wildcard_path, suppress_search_errors=True):
		""" Run's the system's ls command.
		"""
		if suppress_search_errors:
			wildcard_path = wildcard_path + ' 2> /dev/null'
		cmd = '/usr/bin/ls -1 ' + wildcard_path
		with subprocess.Popen(
				cmd, stdout=subprocess.PIPE, shell=False, encoding='utf8') as process:  # nosec
			arr = process.communicate()[0].split('\n')
			items_arr = []
			for file_item in arr:
				if file_item == '':
					continue
				items_arr.append(str(file_item))
			return items_arr

	def cp(self, source_path, destination_path):
		""" Run's the system's cp command.
		"""
		cmd = '/usr/bin/cp -f ' + source_path + ' ' + destination_path
		with subprocess.Popen(
				cmd, stdout=subprocess.PIPE, shell=False, encoding='utf8') as process:  # nosec
			process.communicate()

	def filter_txt_file(self, src_path, dst_path, grep_filters):
		""" Filters a text file by running cat and grep system commands.
		"""
		cmd = (
			'/usr/bin/cat ' + src_path + ' | ' + grep_filters + ' | /usr/bin/tee ' + dst_path +
			' > /dev/null;sleep 5'
		)
		script_loc = '/tmp/filter-script-' + str(random.randint(0, 1000)) + '.sh'  # nosec
		with open(script_loc, 'w', encoding='utf-8') as w:
			w.write(cmd)
		time.sleep(10)
		cmd = '/usr/bin/bash ' + script_loc
		with subprocess.Popen(
				cmd, stdout=subprocess.PIPE, shell=False, encoding='utf8') as process:  # nosec
			process.communicate(timeout=999999)
			time.sleep(10)
			self.remove(script_loc)

	def unzip(self, source_filename, dest_dir):
		""" Run's the python unzip command.
		"""
		with zipfile.ZipFile(source_filename) as zf:
			for member in zf.infolist():
				# Path traversal defense copied from
				# http://hg.python.org/cpython/file/tip/Lib/http/server.py#l789
				words = member.filename.split('/')
				path = dest_dir
				for word in words[:-1]:
					drive, word = os.path.splitdrive(word)
					head, word = os.path.split(word)
					if word in (os.curdir, os.pardir, ''):
						continue
					path = os.path.join(path, word)
				zf.extract(member, path)
		time.sleep(5)
		print('Unzipped  %s to %s' % (source_filename, dest_dir))


class StopWatch():
	"""A multi instance stopwatch to measure computation performance via logging.
	"""
	def __init__(self, title, debug=None, info=None):
		""" Constructor.
		"""
		self.__start = None
		self.__end = None
		self.log = []
		self.__debug = debug
		self.__info = info
		self.__title = title

	def start(self, logmsg=''):
		""" Start the timer.
		"""
		self.__start = time.time()
		logmsg = 'START |%s %s ' % (self.__title, logmsg)
		if self.__debug:
			print('StopWatch: DEBUG> %s' % logmsg)
		if self.__info:
			print('StopWatch: INFO > %s' % logmsg)
		self.log.append(logmsg)

	def end(self, logmsg=''):
		""" End the timer.
		"""
		self.__end = (time.time() - self.__start)
		duration = seconds_human(self.__end)
		logmsg = logmsg.replace('\n', '')
		logmsg = 'END   |%s %s|Duration=%s' % (self.__title, logmsg, duration)
		if self.__debug:
			print('StopWatch: DEBUG> %s' % logmsg)
		if self.__info:
			print('StopWatch: INFO > %s' % logmsg)
		self.log.append(logmsg)

	def get_log(self):
		""" Get the raw log.
		"""
		return self.log

	def print_log(self):
		""" Pretty print the log to standard output.
		"""
		print('StopWatch: Recapitulation...')
		for i in self.log:
			print(i)
		print('StopWatch: Recapitulation...OK')
		print()


class Web():
	""" Tightly coupled to unix systems only. No windows.
		 System curl wrapper only.

		 Example:
			self.API_C = rlcom.Web(
				'command': '/usr/bin/curl',
				'type': 'api-json',
				'headers': '-H "Authorization: Bearer %s"' % self.__API_TOKEN,
				'url': 'https://%s:%s' % (self.__API_HOST, self.__API_PORT)
			)
			account_id = ...
			data = ... dict/json
			r = self.API_C.request(
				url='v3/accounts/%s/orders' % account_id,
				data
			)

	"""
	def __init__(self, **kwargs):
		""" Constructor.
		"""
		if 'command' in kwargs.keys():
			self.CMD = kwargs['command']
		else:
			self.CMD = '/usr/bin/curl'
		if 'headers' in kwargs.keys():
			self.HEADERS = kwargs['headers']
		if 'type' in kwargs.keys():
			self.TYPE = kwargs['type']
		else:
			self.TYPE = 'browser'
		if self.TYPE == 'browser':
			self.CMD = (
				self.CMD + ' --retry 4 --retry-delay 60 --user-agent ' +
				'"Mozilla/5.0 (Windows NT 10.0; rv:78.0) Gecko/20100101 Firefox/78.0"'
			)
		elif self.TYPE == 'api-json':
			self.CMD = (
				self.CMD + ' --retry 4 --retry-delay 60 ' +
				'-H "Content-Type: application/json" ' + self.HEADERS
			)
		if 'url' in kwargs.keys():
			self.URL = kwargs['url']

	def request(self, **kwargs):
		""" INPUT: key value pair of kwargs values
			 OUTPUT: a dictionary -
					{
						'url': <string>,
						'data': <data from kwargs>
					}
		"""
		url = self.URL
		if 'url' in kwargs.keys():
			url = url + '/' + kwargs['url']
		data = kwargs
		data.__delitem__('url')
		return {
			'url': url,
			'data': data
		}

	def put(self, request):
		""" INPUT: request - dictionary - {'url': str, 'data': kwargs}
			 OUTPUT: dictionary - curl string standard output json response
		"""
		try:
			url = request['url']
			data = request['data']
			cmd = self.CMD + ' -X PUT -d "%s" %s' % (data, url)
			with subprocess.Popen(
					cmd, stdout=subprocess.PIPE, shell=False, encoding='utf8') as process:  # nosec
				rtn = process.communicate(timeout=90)[0]
				return dict(rtn)
		except Exception as e:
			print('ERROR: Could not execute unix command ' + self.CMD)
			print(
				'ERROR: Exception for send() ERR - \n' + str(traceback.format_exc()) +
				'\n' + str(e)
			)
			return False

	def post(self, request):
		""" INPUT: request - dictionary - {'url': str, 'data': kwargs}
			 OUTPUT: dictionary - curl string standard output json response
		"""
		try:
			url = request['url']
			data = request['data']
			cmd = self.CMD + ' -X POST -d "%s" %s' % (data, url)
			with subprocess.Popen(
					cmd, stdout=subprocess.PIPE, shell=False, encoding='utf8') as process:  # nosec
				rtn = process.communicate(timeout=90)[0]
				return dict(rtn)
		except Exception as e:
			print('ERROR: Could not execute unix command ' + self.CMD)
			print(
				'ERROR: Exception for send() ERR - \n' + str(traceback.format_exc()) +
				'\n' + str(e)
			)
			return False

	def get(self, request):
		""" INPUT: request - dictionary - {'url': str, 'data': None}
			 OUTPUT: dictionary - curl string standard output json response
		"""
		try:
			url = request['url']
			cmd = self.CMD + ' -X GET %s' % url
			with subprocess.Popen(
					cmd, stdout=subprocess.PIPE, shell=False, encoding='utf8') as process:  # nosec
				rtn = process.communicate(timeout=90)[0]
				return dict(rtn)
		except Exception as e:
			print('ERROR: Could not execute unix command ' + self.CMD)
			print(
				'ERROR: Exception for send() ERR - \n' + str(traceback.format_exc()) +
				'\n' + str(e)
			)
			return False


class VectorizeEngine():
	""" A base set of tools to vectorize a computation and abstract it.
	"""
	def __init__(self, **kwargs):
		""" If no specific library is set, then one will be selected during runtime
			in the below order of precedence. It is best to pre-import a gpu library
			and feed it in via this instance super constructor. This will be faster
			if you have many vectorization layers

			INPUT: **kwargs - set specific gpu / cpu library
							- use_tensorflow_lib=<tf module import object>
							- use_pytorch_lib=<pytorch module import object>
							- use_numpy_lib=<numpy module import object>
		"""
		self.__shift_direction = None
		self.__TF = None
		self.__PT = None
		self.__NP = None
		self.__DROPNA = False
		if 'use_tensorflow_lib' in kwargs.keys():
			self.__TF = kwargs['use_tensorflow_lib']
		elif 'use_pytorch_lib' in kwargs.keys():
			self.__PT = kwargs['use_pytorch_lib']
		elif 'use_numpy_lib' in kwargs.keys():
			self.__NP = kwargs['use_numpy_lib']
		if 'dropna' in kwargs.keys():
			self.__DROPNA = kwargs['dropna']

	def _shiftr(self, data):
		x = data.copy()
		x.insert(0, None)
		x.pop()
		return x

	def _shiftl(self, data):
		x = data.copy()
		x.append(None)
		x.pop(0)
		return x

	def _shift(self, data, x):
		if x > 0:
			d = 1
			e = x
		else:
			d = -1
			e = abs(x)
		if isinstance(data, tuple):
			data = list(data)
		new = data.copy()
		for i in range(0, e, d):
			if d == 1:
				new = self._shiftr(new)
			else:
				new = self._shiftl(new)
		return new

	def _vshift(self, data, start, amount):
		""" Abstracted method to vertically shift the given array N times.
			If N is positive it shifts to the right, else to the left.

			Use this method to take an array input and create a rolling value matrix for
			vectorization.

			INPUT: data - a single array
					start - usually 0
					amount - +/- integer - shift left or right by this much
			OUTPUT: an array of arrays suitable for rolling vectorizations.
		"""
		rtn = [self._shift(data, x) for x in range(start, amount)]
		return tuple(rtn)

	def _dropna(self, data):
		""" Abstracted method to exclude Nones from an array.
		"""
		return [x for x in data if x]

	def _cropna(self, data):
		""" Determines if the array was shifted left or right. Then we make sure the first array has
			no Nones by dropping it. We then cut all the other arrays by the same amount so that it
			is all an evenly cropped square.

			We crop the array matrix at either the left or right side.

			INPUT: array of arrays with values
			OUTPUT: the same as the input but with cropped and squared values based on whether it was
					shifted right or left.
		"""
		data_size = data.__len__()
		is_shifted_left = 0
		is_shifted_right = 0
		for datum in data:
			first = datum[0]
			last = datum[-1]
			if not first:
				is_shifted_right += 1
			elif not last:
				is_shifted_left += 1
		first_array = data[0]
		first_offset_size = 0
		for i in first_array:
			if not i:
				first_offset_size += 1
		cropped = []
		if is_shifted_right > is_shifted_left:
			self.__shift_direction = 1
			for datum in data:
				new = datum[first_offset_size:].copy()
				for a in range(0, data_size - 1):
					new.pop(0)
				cropped.append(new)
		elif is_shifted_left > is_shifted_right:
			self.__shift_direction = -1
			for datum in data:
				new = datum[first_offset_size:].copy()
				for a in range(0, data_size - 1):
					new.pop()
				cropped.append(new)
		elif is_shifted_left == 0 and is_shifted_right == 0:
			return tuple(data)
		else:
			raise ValueError('Bad data structure')
		return tuple(cropped)

	def __square(self, square_size, data):
		""" Helper method for squaring a matrix.
		"""
		new_data = []
		for series in data:
			series_size = series.__len__()
			fill_size = square_size - series_size
			if fill_size == 0:
				new_data.append(list(series))
				continue
			elif square_size < series_size:
				raise ValueError(
					'square_size %s must be larger than the series_size %s' % (
						square_size, series_size
					)
				)
			new_series = [None for x in range(0, fill_size)]
			new_series.extend(list(series))
			new_data.append(new_series)
		return new_data

	def _square_right(self, data):
		""" Pads an array of series so that it is shifted to the right
			and padded with Nones to the left. The result should be a
			squared matrix of values.

			This first array value should have no Nones at the left.

			INPUT:  data    - an array of series data
			OUTPUT: new_data - a squared matrix

		"""
		square_size = 0
		for series in data:
			size = list(series).__len__()
			if size > square_size:
				square_size = size
		# square size is the largest series size
		if square_size == 0:
			raise ValueError('Data needs to be an array of arrays greater than 2.')
		return self.__square(square_size, data)

	def _square_right_by_ref(self, reference, data):
		""" Instead of using the largest series as the reference, we use the specified
			reference for squaring to the right.
		"""
		square_size = list(reference).__len__()
		rtn = None
		if self._is_series(data):
			rtn = self.__square(square_size, [data])[0]
		elif self._is_matrices(data):
			rtn = self.__square(square_size, data)
		else:
			raise ValueError('Bad data type')
		return rtn

	def _has_accel_lib(self, lib='at-least-one'):
		""" INPUT: lib - str - default - all
					- specific lib - tensorflow, pytorch, numpy
			OUTPUT: boolean - True or False
		"""
		has = False
		if lib == 'at-least-one':
			if not self.__TF and not self.__PT and not self.__NP:
				has_tf = self._has_accel_lib(lib='tensorflow')
				has_pt = self._has_accel_lib(lib='pytorch')
				has_np = self._has_accel_lib(lib='numpy')
				if has_tf or has_pt or has_np:
					has = True
				has = False
			elif self.__TF:
				has = True
			elif self.__PT:
				has = True
			elif self.__NP:
				has = True
		elif lib == 'tensorflow' and self.__TF:
			has = True
		elif lib == 'pytorch' and self.__PT:
			has = True
		elif lib == 'numpy' and self.__NP:
			has = True
		return has

	def _is_series(self, data):
		if isinstance(data, list) and data.__len__() == 0:
			return False
		number_types = [int, float, F]
		array_types = [list, tuple, np.array]
		if type(data) in array_types and type(data[0]) not in array_types:
			return True
		return False

	def _is_matrices(self, data):
		if isinstance(data, list) and data.__len__() == 0:
			return False
		number_types = [int, float, F]
		array_types = [list, tuple, np.array]
		if type(data) in array_types and type(data[0]) in array_types and \
				type(data[0][0]) not in array_types:
			return True
		return False

	def _is_a_list_of_series(self, data):
		""" Check if data is a list of series, or a list of matrices.
		"""
		number_types = [int, float]
		array_types = [list, tuple, np.array]
		if type(data[0][0]) in number_types or type(data[0][0]) not in array_types:
			return True
		return False

	def _is_a_list_of_matrices(self, data):
		""" Check if data is a list of series, or a list of matrices.
		"""
		array_types = [list, tuple]
		if type(data[0][0]) in array_types:
			return True
		return False

	def _multiply(self, data):
		""" Multi library mutliply.
		"""
		reduction = None
		if self._has_accel_lib(lib='tensorflow') and self._is_a_list_of_matrices(data):
			tf = self.__TF
			new_data = []
			for matrix in data:
				new_matrix = []
				for mrow in matrix:
					new_matrix.append(tuple(mrow))
				new_data.append(np.array(new_matrix))
			reduction = tf.constant(new_data[0])
			with tf.Session() as sess:
				for matrix in new_data[1:]:
					reduction = tf.matmul(reduction, tf.constant(matrix))
			new_reduction = []
			for matrix in new_reduction:
				new_reduction.append(list(matrix))
			reduction = new_reduction

		elif self._has_accel_lib(lib='tensorflow') and self._is_a_list_of_series(data):
			tf = self.__TF
			reduction = tf.constant(data[0])
			with tf.Session() as sess:
				for layer in data[1:]:
					reduction = sess.run(tf.multiply(reduction, tf.constant(layer)))
			reduction = list(reduction)

		elif self._has_accel_lib(lib='pytorch') and self._is_a_list_of_series(data):
			pt = self.__PT
			reduction = pt.tensor(data[0])
			for layer in data[1:]:
				reduction = pt.multiply(reduction, pt.tensor(layer))
			reduction = list(reduction)

		elif self._has_accel_lib(lib='numpy') and self._is_a_list_of_series(data):
			np = self.__NP
			reduction = np.array(data[0])
			for layer in data[1:]:
				reduction = np.multiply(reduction, np.array(layer))
			reduction = list(reduction)

		else:
			raise RuntimeError('Accellerator library requried.')
		return reduction

	def _add(self, data):
		""" Multi library add
		"""
		reduction = None
		if self._has_accel_lib(lib='tensorflow') and self._is_a_list_of_matrices(data):
			tf = self.__TF
			new_data = []
			for matrix in data:
				new_matrix = []
				for mrow in matrix:
					new_matrix.append(tuple(mrow))
				new_data.append(np.array(new_matrix))
			reduction = tf.constant(new_data[0])
			with tf.Session() as sess:
				for matrix in new_data[1:]:
					reduction = sess.run(tf.add(reduction, tf.constant(matrix)))
			new_reduction = []
			for matrix in new_reduction:
				new_reduction.append(list(matrix))
			reduction = new_reduction

		elif self._has_accel_lib(lib='tensorflow') and self._is_a_list_of_series(data):
			tf = self.__TF
			reduction = tf.constant(data[0])
			with tf.Session() as sess:
				for layer in data[1:]:
					reduction = sess.run(tf.add(reduction, tf.constant(layer)))
			reduction = list(reduction)

		elif self._has_accel_lib(lib='pytorch') and self._is_a_list_of_series(data):
			pt = self.__PT
			reduction = pt.tensor(data[0])
			for layer in data[1:]:
				reduction = pt.add(reduction, pt.tensor(layer))
			reduction = list(reduction)

		elif self._has_accel_lib(lib='numpy') and self._is_a_list_of_series(data):
			np = self.__NP
			reduction = np.array(data[0])
			for layer in data[1:]:
				reduction = np.add(reduction, np.array(layer))
			reduction = list(reduction)

		else:
			raise RuntimeError('Accellerator library requried.')
		return reduction

	def _subtract(self, data):
		""" Multi library subtract
		"""
		reduction = None
		if self._has_accel_lib(lib='tensorflow') and self._is_a_list_of_matrices(data):
			tf = self.__TF
			new_data = []
			for matrix in data:
				new_matrix = []
				for mrow in matrix:
					new_matrix.append(tuple(mrow))
				new_data.append(np.array(new_matrix))
			reduction = tf.constant(new_data[0])
			with tf.Session() as sess:
				for matrix in new_data[1:]:
					reduction = sess.run(tf.subtract(reduction, tf.constant(matrix)))
			new_reduction = []
			for matrix in new_reduction:
				new_reduction.append(list(matrix))
			reduction = new_reduction

		elif self._has_accel_lib(lib='tensorflow') and self._is_a_list_of_series(data):
			tf = self.__TF
			reduction = tf.constant(data[0])
			with tf.Session() as sess:
				for layer in data[1:]:
					reduction = sess.run(tf.subtract(reduction, tf.constant(layer)))
			reduction = list(reduction)

		elif self._has_accel_lib(lib='pytorch') and self._is_a_list_of_series(data):
			pt = self.__PT
			reduction = pt.tensor(data[0])
			for layer in data[1:]:
				reduction = pt.subtract(reduction, pt.tensor(layer))
			reduction = list(reduction)

		elif self._has_accel_lib(lib='numpy') and self._is_a_list_of_series(data):
			np = self.__NP
			reduction = np.array(data[0])
			for layer in data[1:]:
				reduction = reduction - np.array(layer)
			reduction = list(reduction)

		else:
			raise RuntimeError('Accellerator library requried.')
		return reduction

	def _divide(self, data):
		""" Multi library divide
		"""
		reduction = None
		if self._has_accel_lib(lib='tensorflow') and self._is_a_list_of_matrices(data):
			tf = self.__TF
			new_data = []
			for matrix in data:
				new_matrix = []
				for mrow in matrix:
					new_matrix.append(tuple(mrow))
				new_data.append(np.array(new_matrix))
			reduction = tf.constant(new_data[0])
			with tf.Session() as sess:
				for matrix in new_data[1:]:
					reduction = sess.run(tf.divide(reduction, tf.constant(matrix)))
			new_reduction = []
			for matrix in new_reduction:
				new_reduction.append(list(matrix))
			reduction = new_reduction

		elif self._has_accel_lib(lib='tensorflow') and self._is_a_list_of_series(data):
			tf = self.__TF
			reduction = tf.constant(data[0])
			with tf.Session() as sess:
				for layer in data[1:]:
					reduction = sess.run(tf.divide(reduction, tf.constant(layer)))
			reduction = list(reduction)

		elif self._has_accel_lib(lib='pytorch') and self._is_a_list_of_series(data):
			pt = self.__PT
			reduction = pt.tensor(data[0])
			for layer in data[1:]:
				reduction = pt.divide(reduction, pt.tensor(layer))
			reduction = list(reduction)

		elif self._has_accel_lib(lib='numpy') and self._is_a_list_of_series(data):
			np = self.__NP
			reduction = np.array(data[0])
			for layer in data[1:]:
				reduction = reduction / np.array(layer)
			reduction = list(reduction)

		else:
			raise RuntimeError('Accellerator library requried.')
		return reduction

	def _power(self, data):
		""" Multi library power
		"""
		reduction = None
		if self._has_accel_lib(lib='tensorflow') and self._is_a_list_of_matrices(data):
			raise RuntimeError('Power does not currently work with tensorflow')

		elif self._has_accel_lib(lib='tensorflow') and self._is_a_list_of_series(data):
			raise RuntimeError('Power does not currently work with tensorflow')

		elif self._has_accel_lib(lib='pytorch') and self._is_a_list_of_series(data):
			pt = self.__PT
			reduction = pt.tensor(data[0])
			for layer in data[1:]:
				reduction = pt.pow(reduction, pt.tensor(layer))
			reduction = list(reduction)

		elif self._has_accel_lib(lib='numpy') and self._is_a_list_of_series(data):
			np = self.__NP
			reduction = np.array(data[0])
			for layer in data[1:]:
				reduction = np.power(reduction, np.array(layer))
			reduction = list(reduction)

		else:
			raise RuntimeError('Accellerator library requried.')
		return reduction

	def _mode_np_helper(self, x, y, m):
		if x == m:
			return y
		return None

	def _mode_np(self, data):
		np = self.__NP
		cats = 10
		catsa = data.__len__() / 4
		cats = int(min([cats, catsa]))
		n, bins = np.histogram(data, bins=cats, density=True)
		m = max(n)
		bins = list(bins[1:])
		# print(n)
		# print(bins)
		# print(data)
		# print("bins=%s, np_max=%s" % (cats, m))
		found = [self._mode_np_helper(x, y, m) for x, y in zip(n, bins)]
		found = [x for x in found if x]
		# print(found)
		rtn = None
		if found.__len__() > 0:
			max_found = max(found)
			# print("ve_max=%s" % max_found)
			# print()
			rtn = max_found
		return rtn

	def _mode(self, data):
		""" Multi library mode.
			INPUT: an single array
			OUTPUT: float - the most often recurring value
		"""
		mode = None
		if self._has_accel_lib(lib='pytorch') and self._is_a_list_of_series(data):
			pt = self.__PT
			mode = [float(pt.mode(pt.tensor([*x]))[0]) for x in zip(*data)]

		elif self._has_accel_lib(lib='numpy') and self._is_a_list_of_series(data):
			mode = [self._mode_np([*x]) for x in zip(*data)]

		else:
			raise RuntimeError(
				'Accellerator library, numpy or pytorch, requried. Input must be series.'
			)
		return mode

	def __round_series(self, x, round_size):
		rtn = None
		if x != 0 and x is not None:
			rtn = round(float(x), round_size)
		elif x == 0:
			rtn = 0
		return rtn

	def round_series(self, data, round_size):
		""" Round and array's values by round_size number of decimal points.
		"""
		return [self.__round_series(x, round_size) for x in list(data)]

	def final_func(self):
		""" Overrride Me. Mandatory.
		"""
		raise RuntimeError('Override function required.')

	def ignite(self, data, final_func, **kwargs):
		""" Run the vectorized computation as configured by the overriden class instance.
		"""
		if isinstance(data, tuple):
			data = list(data)
		round_size = None
		if 'round_size' in kwargs.keys():
			round_size = kwargs['round_size']
		square_right_ref = data
		if 'square_right_ref' in kwargs.keys():
			square_right_ref = kwargs['square_right_ref']
		massaged_components = []
		for k in kwargs.keys():
			if k == 'row_funcs':
				for f in kwargs[k]:
					massaged_components.append([f(x) for x in data])
			if k == 'columns':
				massaged_components.extend(kwargs[k])
		if massaged_components.__len__() == 0:
			massaged_components = data
		result = final_func(massaged_components)
		if round_size:
			if self._is_series(result):
				result = self.round_series(result, round_size)
			elif self._is_matrices(result):
				new_matrix = []
				for row in result:
					new_matrix.append(self.round_series(row, round_size))
				result = new_matrix
			else:
				raise ValueError('result is not a series or a matrix')
		result_is_m = self._is_matrices(result)
		result_is_s = self._is_series(result)
		is_result_valid = (result_is_s or result_is_m)
		if is_result_valid and not self.__DROPNA:
			# Exploratory result clean ups
			ref = square_right_ref
			if ref.__len__() > 0 and self._is_series(ref[0]) and result_is_s:
				result = self.__square(ref[0].__len__(), [result])[0]
			elif ref.__len__() > 0 and self._is_matrices(ref[0]) and result_is_s:
				result = self.__square(ref[0][0].__len__(), [result])[0]
			else:
				result = self.__square(ref.__len__(), [result])[0]
		if not is_result_valid:
			raise ValueError('Result was not a series or a matrices structure.')
		# Raw output for component mode
		return tuple(result)


def get_direction(array):
	""" Converts list to +1 or -1 depending on whether the values are above or below zero.
	"""
	new = []
	for x in array:
		if x > 0:
			new.append(1)
		elif x < 0:
			new.append(-1)
		else:
			new.append(0)
	return new


def shift_left(array):
	""" Shift list values to the left.
		 Use if you want to see future values.
	"""
	array.append(None)
	return array[1:].copy()


class VRollingSumEngine(VectorizeEngine):
	"""Rolling sum as a vectorized algorithm.
		DESIGN: - Final Engine - Default Configured
					- dupilcate a single row vector array and shift them up to x times
					- sum all vector arrays to a reduced result
				- Component Engine
					- for each pre prepared vector array
						- sum all vector arrays to a reduced result

		INPUT: FINAL ENGINE - DEFAULT
			  kwargs - skip_pre_preparations=False, skip_post_cleaning=False
					 data - array - single value array

			  COMPONENT ENGINE
			  kwargs - skip_pre_preparations=True, skip_post_cleaning=True
					 data - array[array 1..n] - an array of arrays pre prepared

		OUTPUT: array - a reduced vector result
	"""
	def __init__(self, data, window, **kwargs):
		super().__init__(**kwargs)
		self._DATA = data
		self._ROUND_SIZE = 6
		if 'round_size' in kwargs.keys():
			self._ROUND_SIZE = kwargs['round_size']
		self._SKIP_POST_CLEANING = False
		if 'skip_post_cleaning' in kwargs.keys() and kwargs['skip_post_cleaning']:
			self._SKIP_POST_CLEANING = True
			self._ROUND_SIZE = None
		self._SKIP_PRE_PREPARATIONS = False
		if 'skip_pre_preparations' in kwargs.keys() and kwargs['skip_pre_preparations']:
			self._SKIP_PRE_PREPARATIONS = True
		self.__WINDOW = window

	def __sum(self, *args):
		return sum(*args)

	def pure_python_func(self, data):
		""" The pure python implementation of this object
		"""
		return [
			self.__sum(x) for x in zip(*data)
		]

	def final_func(self, data):
		rtn = None
		if self._has_accel_lib():
			rtn = self._add(data)
		rtn = self.pure_python_func(data)
		return rtn

	def get(self):
		""" Invoke abstraction.
		"""
		data = self._DATA
		if self._SKIP_PRE_PREPARATIONS:
			if data.__len__() != self.__WINDOW or not self._is_matrices(data):
				raise ValueError(
					'data array must contain exactly %s arrays' % self.__WINDOW
				)
			x = tuple(data.copy())
		else:
			if data.__len__() != 1 and not self._is_series(data):
				raise ValueError(
					'data arary must be exactly 1 array as a python list type, ' +
					'and contain either a float or int'
				)
			x = self._vshift(data.copy(), 0, self.__WINDOW)
		x = self._cropna(list(x))
		return self.ignite(
			data, self.final_func, columns=list(x), row_funcs=[],
			round_size=self._ROUND_SIZE
		)


def cumsum(data, window, **kwargs):
	""" Invoke abstraction.
	"""
	return VRollingSumEngine(data, window, **kwargs).get()


def sum3(data, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingSumEngine(data, 3, **kwargs).get()


def cumsum3(data, **kwargs):
	""" Alias.
	"""
	return sum3(data, **kwargs)


def sum7(data, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingSumEngine(data, 7, **kwargs).get()


def cumsum7(data, **kwargs):
	""" Alias.
	"""
	return sum7(data, **kwargs)


def sum14(data, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingSumEngine(data, 14, **kwargs).get()


def cumsum14(data, **kwargs):
	""" Alias.
	"""
	return sum14(data, **kwargs)


def sum32(data, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingSumEngine(data, 32, **kwargs).get()


def cumsum32(data, **kwargs):
	""" Alias.
	"""
	return sum32(data, **kwargs)


def sum64(data, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingSumEngine(data, 64, **kwargs).get()


def cumsum64(data, **kwargs):
	""" Alias.
	"""
	return sum64(data, **kwargs)


class VRollingAvgEngine(VectorizeEngine):
	"""Rolling avg as a vectorized algorithm.
		DESIGN: - Final Engine - Default Configured
					- use a rolling sum engine to perform a configurable computation
					- use this engine to perform a pure python vectorized rolling average
				- Component Engine
					- the same as the final engine, except -
						- you must pass in pre prepared colum series for the rolling sum computation

		INPUT: FINAL ENGINE - DEFAULT
			  kwargs - skip_pre_preparations=False, skip_post_cleaning=False
					 data - array - single value array

			  COMPONENT ENGINE
			  kwargs - skip_pre_preparations=True, skip_post_cleaning=True
					 data - array[array 1..n] - an array of arrays pre prepared

		OUTPUT: array - a reduced vector result
	"""
	def __init__(self, data, window, **kwargs):
		super().__init__(**kwargs)
		self._DATA = data
		self._ROUND_SIZE = 6
		if 'round_size' in kwargs.keys():
			self._ROUND_SIZE = kwargs['round_size']
		self._SKIP_POST_CLEANING = False
		if 'skip_post_cleaning' in kwargs.keys() and kwargs['skip_post_cleaning']:
			self._SKIP_POST_CLEANING = True
			self._ROUND_SIZE = None
		self._SKIP_PRE_PREPARATIONS = False
		if 'skip_pre_preparations' in kwargs.keys() and kwargs['skip_pre_preparations']:
			self._SKIP_PRE_PREPARATIONS = True
		self.__VRSUM = VRollingSumEngine(data, window, **kwargs).get
		self.__AVG_WINDOW = window

	def _avg(self, vrsum, divisor):
		f = lambda a, b: a / b
		rtn = [
			f(rsum, total) for rsum, total in zip(
				vrsum, divisor
			)
		]
		return tuple(rtn)

	def pure_python_func(self, data):
		""" The pure python implementation of this object
		"""
		divisor = [self.__AVG_WINDOW for x in range(0, data.__len__())]
		return self._avg(data, divisor)

	def final_func(self, data):
		if self._has_accel_lib():
			raise RuntimeError('Average component must be pure python.')
		return self.pure_python_func(data)

	def get(self):
		""" Invoke abstraction.
		"""
		data = self.__VRSUM()
		data = [x for x in data if x]
		return self.ignite(
			data, self.final_func, columns=[], row_funcs=[],
			round_size=self._ROUND_SIZE, square_right_ref=self._DATA
		)


def avg(data, window, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingAvgEngine(data, window, **kwargs).get()


def avg3(data, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingAvgEngine(data, 3, **kwargs).get()


def avg7(data, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingAvgEngine(data, 7, **kwargs).get()


def avg14(data, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingAvgEngine(data, 14, **kwargs).get()


def avg32(data, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingAvgEngine(data, 32, **kwargs).get()


def avg64(data, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingAvgEngine(data, 64, **kwargs).get()


class VRollingReturnEngine(VectorizeEngine):
	""" Rolling return as a vectorized algorithm.
		Must be raw values that are all positive. No +/- rate of change values!

		DESIGN: - Final Engine - Default Configured
					-
				- Component Engine
					-

		INPUT: FINAL ENGINE - DEFAULT
			  kwargs - skip_pre_preparations=False, skip_post_cleaning=False
					 data - array - single value array
					 window - int - rolling period size
					 uom - D,M,Q,Y - unit of measure - daily, monthly, quarterly, yearly

			  COMPONENT ENGINE
			  kwargs - skip_pre_preparations=True, skip_post_cleaning=True
					 data - array[array 1..n] - an array of arrays pre prepared
					 window - int - rolling period size
					 uom - D,M,Q,Y - unit of measure - daily, monthly, quarterly, yearly
					 geo - ANN, MON, QTR - geometric basis -
							 - annualization, quarterlization, monthlyization

		OUTPUT: array - a reduced vector result
	"""
	def __init__(self, data, window, uom, geo, **kwargs):
		super().__init__(**kwargs)
		self._DATA = data
		self._ROUND_SIZE = 6
		if 'round_size' in kwargs.keys():
			self._ROUND_SIZE = kwargs['round_size']
		self._SKIP_POST_CLEANING = False
		if 'skip_post_cleaning' in kwargs.keys() and kwargs['skip_post_cleaning']:
			self._SKIP_POST_CLEANING = True
			self._ROUND_SIZE = None
		self._SKIP_PRE_PREPARATIONS = False
		if 'skip_pre_preparations' in kwargs.keys() and kwargs['skip_pre_preparations']:
			self._SKIP_PRE_PREPARATIONS = True
		self.__WINDOW = window
		self.__UNIT_OF_MEASURE = None
		if uom == 'D' and geo == 'ANN':
			self.__UNIT_OF_MEASURE = 365
		elif uom == 'M' and geo == 'ANN':
			self.__UNIT_OF_MEASURE = 12
		elif uom == 'Q' and geo == 'ANN':
			self.__UNIT_OF_MEASURE = 4
		elif uom == 'Y' and geo == 'ANN':
			self.__UNIT_OF_MEASURE = 1
		elif uom == 'D' and geo == 'MON':
			self.__UNIT_OF_MEASURE = 31
		elif uom == 'M' and geo == 'MON':
			self.__UNIT_OF_MEASURE = 1
		elif uom == 'D' and geo == 'QTR':
			self.__UNIT_OF_MEASURE = 93
		elif uom == 'M' and geo == 'QTR':
			self.__UNIT_OF_MEASURE = 3
		elif uom == 'Q' and geo == 'QTR':
			self.__UNIT_OF_MEASURE = 1
		else:
			raise ValueError(
				'uom must be either D,M,Q, or Y, with geo basis as either ANN, MON or QTR'
			)

	def __pure_python_func_helper(self, *data):
		C = data[0][0]
		P = data[0][-1]
		S = C - P
		if S == 0:
			return C
		R = S / P
		SR = abs(R)**(float(self.__UNIT_OF_MEASURE) / self.__WINDOW)
		if R < 0:
			SR = SR * -1
		return SR

	def pure_python_func(self, data):
		""" The pure python implementation of this object
		"""
		return [self.__pure_python_func_helper(x) for x in zip(*data)]

	def __accel_func_helper(self, R, STD):
		SR = R**STD
		if R < 0:
			SR = SR * -1
		return SR

	def final_func(self, data):
		rtn = None
		if self._has_accel_lib():
			P = data[-1]
			C = data[0]
			uoms = [self.__UNIT_OF_MEASURE for x in range(0, data[0].__len__())]
			hold = [self.__WINDOW for x in range(0, data[0].__len__())]
			R = self._divide([self._subtract([C, P]), P])
			STD = self._divide([uoms, hold])
			rtn = [self.__accel_func_helper(rtn, std) for rtn, std in zip(R, STD)]
		rtn = self.pure_python_func(data)
		return rtn

	def get(self):
		""" invoke abstraction.
		"""
		data = self._DATA
		if self._SKIP_PRE_PREPARATIONS:
			if data.__len__() != self.__WINDOW or not self._is_matrices(data):
				raise ValueError(
					'data array must contain exactly %s arrays' % self.__WINDOW
				)
			x = tuple(data.copy())
		else:
			if data.__len__() != 1 and not self._is_series(data):
				raise ValueError(
					'data arary must be exactly 1 array as a python list ' +
					'type, and contain either a float or int'
				)
			x = self._vshift(data.copy(), 0, self.__WINDOW)
		x = self._cropna(list(x))
		return self.ignite(
			data, self.final_func, columns=list(x), row_funcs=[],
			round_size=self._ROUND_SIZE
		)


def rtn(data, window, uom, geo, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingReturnEngine(data, window, uom, geo, **kwargs).get()


def rtn3(data, uom, geo, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingReturnEngine(data, 3, uom, geo, **kwargs).get()


def rtn7(data, uom, geo, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingReturnEngine(data, 7, uom, geo, **kwargs).get()


def rtn14(data, uom, geo, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingReturnEngine(data, 14, uom, geo, **kwargs).get()


def rtn32(data, uom, geo, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingReturnEngine(data, 32, uom, geo, **kwargs).get()


def rtn64(data, uom, geo, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingReturnEngine(data, 64, uom, geo, **kwargs).get()


def rtn_8M_D_MON(data, **kwargs):
	""" invoke abstraction.
	"""
	# 8 month hold(248p), daily precision, average monthly realized returns, geometrically normalised
	return rtn(data, 31 * 8, 'D', 'MON', **kwargs)


def rtn_4M_D_MON(data, **kwargs):
	""" invoke abstraction.
	"""
	# 4 month hold(124p), daily precision, average monthly realized returns, geometrically normalised
	return rtn(data, 31 * 4, 'D', 'MON', **kwargs)


def rtn_2M_D_MON(data, **kwargs):
	""" invoke abstraction.
	"""
	# 2 month hold(62p), daily precision, average monthly realized returns, geometrically normalised
	return rtn(data, 31 * 2, 'D', 'MON', **kwargs)


def rtn_2Q_D_QTR(data, **kwargs):
	""" invoke abstraction.
	"""
	# 2 quarter hold(186p), daily precision, average quarterly realized returns
	return rtn(data, 31 * 3 * 2, 'D', 'QTR', **kwargs)


def rtn_2Y_D_ANN(data, **kwargs):
	""" invoke abstraction.
	"""
	# 2 year hold(744p), daily precision, average annual realized returns
	return rtn(data, 31 * 12 * 2, 'D', 'ANN', **kwargs)


class VRollingMinEngine(VectorizeEngine):
	""" Rolling min as a vectorized algorithm.
		DESIGN: - Final Engine - Default Configured
					-
				- Component Engine
					-

		INPUT: FINAL ENGINE - DEFAULT
			  kwargs - skip_pre_preparations=False, skip_post_cleaning=False
					 data - array - single value array

			  COMPONENT ENGINE
			  kwargs - skip_pre_preparations=True, skip_post_cleaning=True
					 data - array[array 1..n] - an array of arrays pre prepared

		OUTPUT: array - a reduced vector result
	"""
	def __init__(self, data, window, **kwargs):
		super().__init__(**kwargs)
		self._DATA = data
		self._ROUND_SIZE = 6
		if 'round_size' in kwargs.keys():
			self._ROUND_SIZE = kwargs['round_size']
		self._SKIP_POST_CLEANING = False
		if 'skip_post_cleaning' in kwargs.keys() and kwargs['skip_post_cleaning']:
			self._SKIP_POST_CLEANING = True
			self._ROUND_SIZE = None
		self._SKIP_PRE_PREPARATIONS = False
		if 'skip_pre_preparations' in kwargs.keys() and kwargs['skip_pre_preparations']:
			self._SKIP_PRE_PREPARATIONS = True
		self.__WINDOW = window

	def pure_python_func(self, data):
		""" The pure python implementation of this object
		"""
		f = lambda *args: min(*args)
		return [
			f(x) for x in zip(*data)
		]

	def final_func(self, data):
		if self._has_accel_lib():
			raise RuntimeError('Min component must be pure python.')
		else:
			return self.pure_python_func(data)

	def get(self):
		""" invoke abstraction.
		"""
		data = self._DATA
		if self._SKIP_PRE_PREPARATIONS:
			if data.__len__() != self.__WINDOW or not self._is_matrices(data):
				raise ValueError(
					'data array must contain exactly %s arrays' % self.__WINDOW
				)
			x = tuple(data.copy())
		else:
			if data.__len__() != 1 and not self._is_series(data):
				raise ValueError(
					'data arary must be exactly 1 array as a python list ' +
					'type, and contain either a float or int'
				)
			x = self._vshift(data.copy(), 0, self.__WINDOW)
		x = self._cropna(list(x))
		return self.ignite(
			data, self.final_func, columns=list(x), row_funcs=[],
			round_size=self._ROUND_SIZE
		)


def minr(data, window, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingMinEngine(data, window, **kwargs).get()


def min3(data, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingMinEngine(data, 3, **kwargs).get()


def min7(data, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingMinEngine(data, 7, **kwargs).get()


def min14(data, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingMinEngine(data, 14, **kwargs).get()


def min32(data, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingMinEngine(data, 32, **kwargs).get()


def min64(data, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingMinEngine(data, 64, **kwargs).get()


class VRollingMaxEngine(VectorizeEngine):
	""" Rolling max as a vectorized algorithm.
		DESIGN: - Final Engine - Default Configured
					-
				- Component Engine
					-

		INPUT: FINAL ENGINE - DEFAULT
			  kwargs - skip_pre_preparations=False, skip_post_cleaning=False
					 data - array - single value array

			  COMPONENT ENGINE
			  kwargs - skip_pre_preparations=True, skip_post_cleaning=True
					 data - array[array 1..n] - an array of arrays pre prepared

		OUTPUT: array - a reduced vector result
	"""
	def __init__(self, data, window, **kwargs):
		super().__init__(**kwargs)
		self._DATA = data
		self._ROUND_SIZE = 6
		if 'round_size' in kwargs.keys():
			self._ROUND_SIZE = kwargs['round_size']
		self._SKIP_POST_CLEANING = False
		if 'skip_post_cleaning' in kwargs.keys() and kwargs['skip_post_cleaning']:
			self._SKIP_POST_CLEANING = True
			self._ROUND_SIZE = None
		self._SKIP_PRE_PREPARATIONS = False
		if 'skip_pre_preparations' in kwargs.keys() and kwargs['skip_pre_preparations']:
			self._SKIP_PRE_PREPARATIONS = True
		self.__WINDOW = window

	def pure_python_func(self, data):
		""" The pure python implementation of this object
		"""
		f = lambda *args: max(*args)
		return [
			f(x) for x in zip(*data)
		]

	def final_func(self, data):
		if self._has_accel_lib():
			raise RuntimeError('Max component must be pure python.')
		else:
			return self.pure_python_func(data)

	def get(self):
		""" invoke abstraction.
		"""
		data = self._DATA
		if self._SKIP_PRE_PREPARATIONS:
			if data.__len__() != self.__WINDOW or not self._is_matrices(data):
				raise ValueError(
					'data array must contain exactly %s arrays' % self.__WINDOW
				)
			x = tuple(data.copy())
		else:
			if data.__len__() != 1 and not self._is_series(data):
				raise ValueError(
					'data arary must be exactly 1 array as a python list ' +
					'type, and contain either a float or int'
				)
			x = self._vshift(data.copy(), 0, self.__WINDOW)
		x = self._cropna(list(x))
		return self.ignite(
			data, self.final_func, columns=list(x), row_funcs=[],
			round_size=self._ROUND_SIZE
		)


def maxr(data, window, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingMaxEngine(data, window, **kwargs).get()


def max3(data, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingMaxEngine(data, 3, **kwargs).get()


def max7(data, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingMaxEngine(data, 7, **kwargs).get()


def max14(data, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingMaxEngine(data, 14, **kwargs).get()


def max32(data, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingMaxEngine(data, 32, **kwargs).get()


def max64(data, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingMaxEngine(data, 64, **kwargs).get()


class VRollingModeEngine(VectorizeEngine):
	"""Rolling mode as a vectorized algorithm.
		DESIGN: - Final Engine - Default Configured
					- use np.histogram or pt.mode functions to determine most often recuring values
					- both libraries will yeild difference results because they are similar in
						implementation
					- we dot not care about exactness
					- numpy is defaulted to 10 bins if window is equal to or larger than 40
					- pytorch has its own method to yield mode
					- numpy is more reactive to rolling fluctuations
				- Component Engine
					- the same as the final engine, except -
						- you must pass in pre prepared colum series for the rolling sum computation

		INPUT: FINAL ENGINE - DEFAULT
			  kwargs - skip_pre_preparations=False, skip_post_cleaning=False
					 data - array - single value array

			  COMPONENT ENGINE
			  kwargs - skip_pre_preparations=True, skip_post_cleaning=True
					 data - array[array 1..n] - an array of arrays pre prepared

		OUTPUT: array - a reduced vector result
	"""
	def __init__(self, data, window, **kwargs):
		super().__init__(**kwargs)
		self._DATA = data
		self._ROUND_SIZE = 2
		if 'round_size' in kwargs.keys():
			self._ROUND_SIZE = kwargs['round_size']
		self._SKIP_POST_CLEANING = False
		if 'skip_post_cleaning' in kwargs.keys() and kwargs['skip_post_cleaning']:
			self._SKIP_POST_CLEANING = True
			self._ROUND_SIZE = None
		self._SKIP_PRE_PREPARATIONS = False
		if 'skip_pre_preparations' in kwargs.keys() and kwargs['skip_pre_preparations']:
			self._SKIP_PRE_PREPARATIONS = True
		self.__WINDOW = window

	def final_func(self, data):
		if self._has_accel_lib(lib='numpy') or self._has_accel_lib(lib='pytorch'):
			return self._mode(data)
		raise RuntimeError('Mode component must use numpy or pytorch.')

	def get(self):
		""" invoke abstraction.
		"""
		data = self._DATA
		if self._SKIP_PRE_PREPARATIONS:
			if data.__len__() != self.__WINDOW or not self._is_matrices(data):
				raise ValueError(
					'data array must contain exactly %s arrays' % self.__WINDOW
				)
			x = tuple(data.copy())
		else:
			if data.__len__() != 1 and not self._is_series(data):
				raise ValueError(
					'data arary must be exactly 1 array as a python list ' +
					'type, and contain either a float or int'
				)
			x = self._vshift(data.copy(), 0, self.__WINDOW)
		x = self._cropna(list(x))
		return self.ignite(
			data, self.final_func, columns=list(x), row_funcs=[],
			round_size=self._ROUND_SIZE
		)


def mode(data, window, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingModeEngine(data, window, **kwargs).get()


def mode3(data, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingModeEngine(data, 3, **kwargs).get()


def mode7(data, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingModeEngine(data, 7, **kwargs).get()


def mode14(data, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingModeEngine(data, 14, **kwargs).get()


def mode32(data, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingModeEngine(data, 32, **kwargs).get()


def mode64(data, **kwargs):
	""" invoke abstraction.
	"""
	return VRollingModeEngine(data, 64, **kwargs).get()


class VGrowthRateEngine(VectorizeEngine):
	"""Growth rate as a vectorized algorithm.
		DESIGN: - Final Engine - Default Configured
					-
				- Component Engine
					-

		INPUT: FINAL ENGINE - DEFAULT
			  kwargs - skip_pre_preparations=False, skip_post_cleaning=False
					 data - array - single value array

			  COMPONENT ENGINE
			  kwargs - skip_pre_preparations=True, skip_post_cleaning=True
					 data - array[array 1..n] - an array of arrays pre prepared

		OUTPUT: array - a reduced vector result
	"""
	def __init__(self, data, **kwargs):
		super().__init__(**kwargs)
		self._DATA = data
		self._ROUND_SIZE = 6
		if 'round_size' in kwargs.keys():
			self._ROUND_SIZE = kwargs['round_size']
		self._SKIP_POST_CLEANING = False
		if 'skip_post_cleaning' in kwargs.keys() and kwargs['skip_post_cleaning']:
			self._SKIP_POST_CLEANING = True
			self._ROUND_SIZE = None
		self._SKIP_PRE_PREPARATIONS = False
		if 'skip_pre_preparations' in kwargs.keys() and kwargs['skip_pre_preparations']:
			self._SKIP_PRE_PREPARATIONS = True
		self.__SHIFTS = 2

	def pure_python_func(self, data):
		""" The pure python implementation of this object
		"""
		f = lambda *args: (args[0][0] - args[0][1]) / args[0][1]
		return [f(x) for x in zip(*data)]

	def final_func(self, data):
		if self._has_accel_lib():
			c, p = data[0], data[1]
			return self._divide([self._subtract([c, p]), p])
		return self.pure_python_func(data)

	def get(self):
		""" invoke abstraction.
		"""
		data = self._DATA
		if self._SKIP_PRE_PREPARATIONS:
			if data.__len__() != self.__SHIFTS or not self._is_matrices(data):
				raise ValueError(
					'data array must contain exactly %s arrays' % self.__SHIFTS
				)
			x = tuple(data.copy())
		else:
			if data.__len__() != 1 and not self._is_series(data):
				raise ValueError(
					'data arary must be exactly 1 array as a python list ' +
					'type, and contain either a float or int'
				)
			x = self._vshift(data.copy(), 0, self.__SHIFTS)
			x = self._cropna(list(x))
		return self.ignite(
			data, self.final_func, columns=list(x), row_funcs=[],
			round_size=self._ROUND_SIZE
		)


def gr(data, **kwargs):
	""" invoke abstraction.
	"""
	return VGrowthRateEngine(data, **kwargs).get()


class VVarianceEngine(VectorizeEngine):
	"""Rolling variance as a vectorized algorithm.
		DESIGN: - Final Engine - Default Configured
					- use a rolling sum engine to perform a configurable computation
					- use this engine to perform a pure python vectorized rolling average
				- Component Engine
					- the same as the final engine, except -
						- you must pass in pre prepared colum series for the rolling sum computation

		INPUT: FINAL ENGINE - DEFAULT
			  kwargs - skip_pre_preparations=False, skip_post_cleaning=False
					 data - array - single value array

			  COMPONENT ENGINE
			  kwargs - skip_pre_preparations=True, skip_post_cleaning=True
					 data - array[array 1..n] - an array of arrays pre prepared

		OUTPUT: array - a reduced vector result
	"""
	def __init__(self, data, window, **kwargs):
		super().__init__(**kwargs)
		self._DATA = data
		self._ROUND_SIZE = 6
		if 'round_size' in kwargs.keys():
			self._ROUND_SIZE = kwargs['round_size']
		self._SKIP_POST_CLEANING = False
		if 'skip_post_cleaning' in kwargs.keys() and kwargs['skip_post_cleaning']:
			self._SKIP_POST_CLEANING = True
			self._ROUND_SIZE = None
		self._SKIP_PRE_PREPARATIONS = False
		if 'skip_pre_preparations' in kwargs.keys() and kwargs['skip_pre_preparations']:
			self._SKIP_PRE_PREPARATIONS = True
		self.__WINDOW = window

	def pure_python_func(self, data):
		""" The pure python implementation of this object
		"""
		fS, fS2 = lambda *x: sum(*x), lambda x: sum([x**2 for x in x])
		fV = lambda *x: (fS2(*x) - (fS(*x)**2 / self.__WINDOW)) / (self.__WINDOW - 1)
		return [fV(x) for x in zip(*data)]

	def final_func(self, data):
		if self._has_accel_lib():
			# raise RuntimeError('Variance component must be pure python.')
			fS2 = lambda x: sum([x**2 for x in x])
			ones = [1 for x in range(0, data[0].__len__())]
			twos = [2 for x in range(0, data[0].__len__())]
			ntotal = [self.__WINDOW for x in range(0, data[0].__len__())]
			return self._divide([
				self._subtract([
					[fS2(x) for x in zip(*data)],
					self._divide([
						self._power([
							self._add(data),
							twos
						]),
						ntotal
					])
				]),
				self._subtract([ntotal, ones])
			])
		return self.pure_python_func(data)

	def get(self):
		""" invoke abstraction.
		"""
		data = self._DATA
		if self._SKIP_PRE_PREPARATIONS:
			if data.__len__() != self.__WINDOW or not self._is_matrices(data):
				raise ValueError(
					'data array must contain exactly %s arrays' % self.__WINDOW
				)
			x = tuple(data.copy())
		else:
			if data.__len__() != 1 and not self._is_series(data):
				raise ValueError(
					'data arary must be exactly 1 array as a python list ' +
					'type, and contain either a float or int'
				)
			x = self._vshift(data.copy(), 0, self.__WINDOW)
		x = self._cropna(list(x))
		return self.ignite(
			data, self.final_func, columns=list(x), row_funcs=[],
			round_size=self._ROUND_SIZE
		)


def var(data, window, **kwargs):
	""" invoke abstraction.
	"""
	return VVarianceEngine(data, window, **kwargs).get()


def var3(data, **kwargs):
	""" invoke abstraction.
	"""
	return VVarianceEngine(data, 3, **kwargs).get()


def var7(data, **kwargs):
	""" invoke abstraction.
	"""
	return VVarianceEngine(data, 7, **kwargs).get()


def var14(data, **kwargs):
	""" invoke abstraction.
	"""
	return VVarianceEngine(data, 14, **kwargs).get()


def var32(data, **kwargs):
	""" invoke abstraction.
	"""
	return VVarianceEngine(data, 32, **kwargs).get()


def var64(data, **kwargs):
	""" invoke abstraction.
	"""
	return VVarianceEngine(data, 64, **kwargs).get()


class VStandardDeviationEngine(VectorizeEngine):
	"""Rolling standard deviation as a vectorized algorithm.
		DESIGN: - Final Engine - Default Configured
					- use a rolling sum engine to perform a configurable computation
					- use this engine to perform a pure python vectorized rolling average
				- Component Engine
					- the same as the final engine, except -
						- you must pass in pre prepared colum series for the rolling sum computation

		INPUT: FINAL ENGINE - DEFAULT
			  kwargs - skip_pre_preparations=False, skip_post_cleaning=False
					 data - array - single value array

			  COMPONENT ENGINE
			  kwargs - skip_pre_preparations=True, skip_post_cleaning=True
					 data - array[array 1..n] - an array of arrays pre prepared

		OUTPUT: array - a reduced vector result
	"""
	def __init__(self, data, window, **kwargs):
		super().__init__(**kwargs)
		self._DATA = data
		self._ROUND_SIZE = 6
		if 'round_size' in kwargs.keys():
			self._ROUND_SIZE = kwargs['round_size']
			del kwargs['round_size']
		kwargs.update({'round_size': None})
		self._SKIP_POST_CLEANING = False
		if 'skip_post_cleaning' in kwargs.keys() and kwargs['skip_post_cleaning']:
			self._SKIP_POST_CLEANING = True
			self._ROUND_SIZE = None
		self._SKIP_PRE_PREPARATIONS = False
		if 'skip_pre_preparations' in kwargs.keys() and kwargs['skip_pre_preparations']:
			self._SKIP_PRE_PREPARATIONS = True
		self.__VRVAR = VVarianceEngine(data, window, **kwargs).get
		self.__WINDOW = window

	def _std(self, data):
		V = data
		return [math.sqrt(x) for x in V if x]

	def pure_python_func(self, data):
		""" The pure python implementation of this object
		"""
		return self._std(data)

	def final_func(self, data):
		return self.pure_python_func(data)

	def get(self):
		""" invoke abstraction.
		"""
		data = [x for x in self.__VRVAR()]
		return self.ignite(
			data, self.final_func, columns=[], row_funcs=[],
			round_size=self._ROUND_SIZE, square_right_ref=self._DATA
		)


def std(data, window, **kwargs):
	""" invoke abstraction.
	"""
	return VStandardDeviationEngine(data, window, **kwargs).get()


def std3(data, **kwargs):
	""" invoke abstraction.
	"""
	return VStandardDeviationEngine(data, 3, **kwargs).get()


def std7(data, **kwargs):
	""" invoke abstraction.
	"""
	return VStandardDeviationEngine(data, 7, **kwargs).get()


def std14(data, **kwargs):
	""" invoke abstraction.
	"""
	return VStandardDeviationEngine(data, 14, **kwargs).get()


def std32(data, **kwargs):
	""" invoke abstraction.
	"""
	return VStandardDeviationEngine(data, 32, **kwargs).get()


def std64(data, **kwargs):
	""" invoke abstraction.
	"""
	return VStandardDeviationEngine(data, 64, **kwargs).get()


class OddsTools():
	""" Methods to help with constructing odds related vectorizations.
	"""
	def __get_fall(self, x):
		if x < 0:
			return 1
		return 0

	def __get_rise(self, x):
		if x > 0:
			return 1
		return 0

	def get_growth_rate_as_tagged_tuple(self, series):
		""" Use for vectorisation computations. A pre data format filter.

			INPUT: array of growth rate values.
			OUTPUT: a tag of growth rate values
					a tuple representing rise and fall for vectorisation computations
					(
						rise as value 1 other is 0,
						fall as value 1 other is 0
					)
		"""
		series_list = list(series)
		series_size = series_list.__len__()
		new = [(self.__get_rise(x), self.__get_fall(x)) for x in series_list if x]
		dropped = series_size - new.__len__()
		if dropped == 0:
			return tuple(new)
		return new


tag = OddsTools().get_growth_rate_as_tagged_tuple


class VDirectionalBayesianMatrixEngine(VectorizeEngine):
	"""A 4 by 4 rolling vecotrized bayesian inference matrix ideal for time series inference.
		Use it for deductive lock step, co-incidence, and seasonality.

		DESIGN: - Final Engine - Default Configured
					-
				- Component Engine
					- the same as the final engine, except -
						- you must pass in pre prepared colum series for the rolling sum computation

		INPUT: FINAL ENGINE - DEFAULT
			  kwargs - skip_pre_preparations=False, skip_post_cleaning=False
					 data - array[[speculation grs], [benchmark grs]]
						  - input series must be raw growth rate value +/- values

			  COMPONENT ENGINE
			  kwargs - skip_pre_preparations=True, skip_post_cleaning=True
					 data - array[
								 [[speculation tagged tuples window],1...n],
								 [[benchmark tagged tuples window],1...n]
							]
						  - pre prepared
						  - input series must be pre converted into a series of tuple tags

			  Generic Inference
					Evidence_tag - Evidence series in tagged form. E1 & E2 - Rise & Fall.
					Hypotheses_tag - Hypotheses series in tagged form. H1 & H2 - Rise & Fall.

			  Generic Growth Rate Inference
					SPtag - Speculation growth rate - wraps Evidence_tag
					BMtag - Benchmark growth rate - wraps Hypotheses_tag

		OUTPUT:
			  Tuple of arrays containing the rolling posterior odds, given the sp and bm inputs.
				  - (
					  [P(H1|E1), P(H2|E1), P(H1|E2), P(H2|E2)],
					  ...
					)

			  Generic Inference
					H1_E1 - P(H1|E1)
					H2_E2 - P(H2|E2)
					H2_E1 - P(H2|E1)
					H1_E2 - P(H1|E2)

			  Generic Growth Rate Inference
					BMrise_SPrise - wraps H1_E1
					BMfall_SPfall - wraps H2_E2
					BMfall_SPrise - wraps H2_E1
					BMrise_SPfall - wraps H1_E2

	"""
	def __init__(self, data, window, **kwargs):
		super().__init__(**kwargs)
		self._DATA = data
		self._ROUND_SIZE = 6
		if 'round_size' in kwargs.keys():
			self._ROUND_SIZE = kwargs['round_size']
		self._SKIP_POST_CLEANING = False
		if 'skip_post_cleaning' in kwargs.keys() and kwargs['skip_post_cleaning']:
			self._SKIP_POST_CLEANING = True
			self._ROUND_SIZE = None
		self._SKIP_PRE_PREPARATIONS = False
		if 'skip_pre_preparations' in kwargs.keys() and kwargs['skip_pre_preparations']:
			self._SKIP_PRE_PREPARATIONS = True
		self.__WINDOW = window

	def __min_zero(self, value):
		if value == 0:
			return 1
		return value

	def pure_python_func(self, SP_rolling, BM_rolling):
		""" The pure python implementation of this object
		"""
		# Calculate Prior
		#  - Create components
		count_ones = lambda x: [i for i in x if i == 1].__len__()
		rise = lambda x: [i[0] for i in x]
		fall = lambda x: [i[1] for i in x]
		sp_r = [rise(x) for x in zip(*SP_rolling)]
		sp_f = [fall(x) for x in zip(*SP_rolling)]
		bm_r = [rise(x) for x in zip(*BM_rolling)]
		bm_f = [fall(x) for x in zip(*BM_rolling)]
		intersect = lambda s, b: [x * y for x, y in zip(s, b)]
		#  - Reduce into BM quadrant components
		#  - multiply the rolling rises and falls
		p_h1_e1 = [intersect(s, b) for s, b in zip(sp_r, bm_r)]
		p_h2_e2 = [intersect(s, b) for s, b in zip(sp_f, bm_f)]
		# - count the rolling ones
		bm_p_h1_e1 = [count_ones(x) for x in p_h1_e1]
		bm_p_h2_e2 = [count_ones(x) for x in p_h2_e2]
		#  - multiply the rolling rises and falls
		p_h1_e2 = [intersect(s, b) for s, b in zip(sp_f, bm_r)]
		p_h2_e1 = [intersect(s, b) for s, b in zip(sp_r, bm_f)]
		# - count the rolling ones
		bm_p_h1_e2 = [count_ones(x) for x in p_h1_e2]
		bm_p_h2_e1 = [count_ones(x) for x in p_h2_e1]

		# Calculate Posterior
		f = lambda x: (
			F(self.__min_zero(x[0]) / self.__min_zero(x[1])),  # P(H1|E1) / P(H2|E1)
			F(self.__min_zero(x[1]) / self.__min_zero(x[0])),  # P(H2|E1) / P(H1|E1)
			F(self.__min_zero(x[2]) / self.__min_zero(x[3])),  # P(H1|E2) / P(H2|E2)
			F(self.__min_zero(x[3]) / self.__min_zero(x[2]))   # P(H2|E2) / P(H1|E2)
		)
		# - left to right - top to bottom - matrix
		bm = [
			f(x) for x in zip(
				bm_p_h1_e1, bm_p_h2_e1,
				bm_p_h1_e2, bm_p_h2_e2
			)
		]
		return bm

	def final_func(self, data):
		SP = data[0]
		BM = data[1]
		if self._has_accel_lib():
			raise RuntimeError('BayesianMatrix component must be pure python.')
		else:
			return self.pure_python_func(SP, BM)

	def get_rolling_data(self):
		""" Loosely coupled data shaping logic.
		"""
		data = self._DATA
		if data.__len__() != 2 or not self._is_series(data[0]):
			raise ValueError(
				'data array must contain exactly 2 arrays containing series ' +
				'growth rate values +/-'
			)
		sp = self._vshift(tag(gr(list(data[0]).copy())), 0, self.__WINDOW)
		bm = self._vshift(tag(gr(list(data[1]).copy())), 0, self.__WINDOW)
		return sp, bm

	def get(self):
		""" Invoke abstraction.
		"""
		data = self._DATA
		ref = None
		if self._SKIP_PRE_PREPARATIONS:
			if data.__len__() != 2 or data[0].__len__() != self.__WINDOW:
				raise ValueError(
					'data array must contain exactly 2 arrays containing rolling ' +
					'window arrays of %s values' % self.__WINDOW
				)
			ref = self._DATA[1][0]
			sp = tuple(data[0])
			bm = tuple(data[1])
		else:
			ref = self._DATA[1]
			sp, bm = self.get_rolling_data()
		sp, bm = self._cropna(sp), self._cropna(bm)
		return self.ignite(
			data, self.final_func, columns=[sp, bm], row_funcs=[],
			round_size=self._ROUND_SIZE, square_right_ref=ref
		)


def bme(data, window, **kwargs):
	""" Invoke abstraction.
	"""
	return VDirectionalBayesianMatrixEngine(data, window, **kwargs).get()


def bme3(data, **kwargs):
	""" Invoke abstraction.
	"""
	return VDirectionalBayesianMatrixEngine(data, 3, **kwargs).get()


def bme7(data, **kwargs):
	""" Invoke abstraction.
	"""
	return VDirectionalBayesianMatrixEngine(data, 7, **kwargs).get()


def bme14(data, **kwargs):
	""" Invoke abstraction.
	"""
	return VDirectionalBayesianMatrixEngine(data, 14, **kwargs).get()


def bme32(data, **kwargs):
	""" Invoke abstraction.
	"""
	return VDirectionalBayesianMatrixEngine(data, 32, **kwargs).get()


def bme64(data, **kwargs):
	""" Invoke abstraction.
	"""
	return VDirectionalBayesianMatrixEngine(data, 64, **kwargs).get()


class VLockStepBayesianMatrixEngine(VDirectionalBayesianMatrixEngine):
	"""Rolling lock step bayesian matrix as a vectorized algorithm.

		DESIGN: - Final Engine - Default Configured
					-
				- Component Engine
					- the same as the final engine, except -
						- you must pass in pre prepared colum series for the rolling sum computation
				-

		INPUT: FINAL ENGINE - DEFAULT
			  kwargs - skip_pre_preparations=False, skip_post_cleaning=False
					 data - array[[speculation grs], [benchmark grs]]
						  - input series must be raw growth rate value +/- values

			  COMPONENT ENGINE
			  kwargs - skip_pre_preparations=True, skip_post_cleaning=True
					 data - array[
								 [[speculation tagged tuples window],1...n],
								 [[benchmark tagged tuples window],1...n]
							]
						  - pre prepared
						  - input series must be pre converted into a series of tuple tags

			  Generic Inference
					Evidence_tag - Evidence series in tagged form. E1 & E2 - Rise & Fall.
					Hypotheses_tag - Hypotheses series in tagged form. H1 & H2 - Rise & Fall.

			  Generic Growth Rate Inference
					SPtag - Speculation growth rate - wraps Evidence_tag
					BMtag - Benchmark growth rate - wraps Hypotheses_tag
	"""
	def __init__(self, data, window, lag, **kwargs):
		super().__init__(data, window, **kwargs)
		self.__WINDOW = window
		if lag <= 0:
			raise ValueError('Lag must be 1 or more for lock step calculations.')
		self.__LAG = lag

	def get_rolling_data(self):
		""" Loosely coupled data shaping logic.
		"""
		data = self._DATA
		if data.__len__() != 2 or not self._is_series(data[0]):
			raise ValueError(
				'data array must contain exactly 2 arrays containing series ' +
				'growth rate values +/-'
			)
		# Co-incide the previous value for bayesian matrix intersection calculations, ie. lock step
		dropna = self._dropna
		if self.__LAG == 1:
			sp = dropna(data[0])
			sp = self._vshift(tag(gr(list(sp).copy())), 0, self.__WINDOW)

		elif self.__LAG > 1:
			sp = self._dropna(data[0])
			sp = self._vshift(
				tag(
					dropna(
						cumsum(
							dropna(
								gr(sp)
							),
							self.__LAG
						)
					)
				),
				0,
				self.__WINDOW
			)

		bm = self._vshift(tag(gr(list(data[1]).copy())), 0, self.__WINDOW)
		rtn = self._square_right([sp, bm])
		return rtn[0], rtn[1]

	def get(self):
		""" Invoke abstraction.
		"""
		data = self._DATA
		if self._SKIP_PRE_PREPARATIONS:
			if data.__len__() != 2 or data[0].__len__() != self.__WINDOW:
				raise ValueError(
					'data array must contain exactly 2 arrays containing rolling ' +
					'window arrays of %s values' % self.__WINDOW
				)
			ref = self._DATA[1][0]
			sp = tuple(data[0].copy())
			bm = tuple(data[1].copy())
		else:
			ref = self._DATA[1]
			sp, bm = self.get_rolling_data()
		sp, bm = self._cropna(sp), self._cropna(bm)
		return self.ignite(
			data, self.final_func, columns=[sp, bm], row_funcs=[],
			round_size=self._ROUND_SIZE, square_right_ref=ref
		)


def bme_lockstep(data, window, lag, **kwargs):
	""" Invoke abstraction.
	"""
	return VLockStepBayesianMatrixEngine(data, window, lag, **kwargs).get()


def bme_lockstep3(data, lag, **kwargs):
	""" Invoke abstraction.
	"""
	return VLockStepBayesianMatrixEngine(data, 3, lag, **kwargs).get()


def bme_lockstep7(data, lag, **kwargs):
	""" Invoke abstraction.
	"""
	return VLockStepBayesianMatrixEngine(data, 7, lag, **kwargs).get()


def bme_lockstep14(data, lag, **kwargs):
	""" Invoke abstraction.
	"""
	return VLockStepBayesianMatrixEngine(data, 14, lag, **kwargs).get()


def bme_lockstep32(data, lag, **kwargs):
	""" Invoke abstraction.
	"""
	return VLockStepBayesianMatrixEngine(data, 32, lag, **kwargs).get()


def bme_lockstep64(data, lag, **kwargs):
	""" Invoke abstraction.
	"""
	return VLockStepBayesianMatrixEngine(data, 64, lag, **kwargs).get()
