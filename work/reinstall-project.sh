#!/usr/bin/bash
cwdf=$(realpath $0)
cwdd=$(dirname ${cwdf})

SP_DIR=${cwdd}
PHOME=~/Projects
LOGD=$SP_DIR/logs
LOGF=$LOGD/$(date +"%Y-%m-%d")-install.out
rm -f $LOGF
find $SP_DIR/src/lib -name '__pycache__' -type d -print0 | xargs --null rm -fr
find $SP_DIR/src/lib -name '*.pyc' -type f -print0 | xargs --null rm -f
export PATH=$PATH:~/.local/bin
f_reinstall () {
	echo
	echo
	echo
	echo
	echo 'Checking Code Style and Errors...'
	echo 'Reinstalling '$SP_DIR'...'
	cd $SP_DIR
	find $SP_DIR/src/lib -maxdepth 1 -mindepth 1 -name '[a-Z]*.py' -type f -print0 | xargs --null flake8 --show-source --doctests --max-line-length 100 --indent-size 4 --ignore E712,W504,E902,F841,F823,E731
	if [ $? -ne 0 ]; then exit 1; fi
	if [ $? -ne 0 ]; then exit 1; fi
	echo 'Flake8 - PASSED'
	find $SP_DIR/src/lib -maxdepth 1 -mindepth 1 -name '[a-Z]*.py' -type f -print0 | xargs --null /usr/bin/pylint --rcfile $SP_DIR/pylint.conf
	if [ $? -ne 0 ]; then exit 1; fi
	echo 'Pylint - PASSED'
	echo 'Generating UML Diagrams...'
	pyreverse -o jpg $SP_DIR/src/lib
	if [ $? -ne 0 ]
	then
		echo 'Pyreverse needs graphviz installed. Skipping. FAILED.'
	else
		echo 'Pyreverse - UMLs generated - PASSED'
	fi
	echo 'Checking Security Issues...'
	bandit -c $SP_DIR/bandit.conf -r $SP_DIR/src/lib
	if [ $? -ne 0 ]; then exit 1; fi
	grep -r '# nosec' $SP_DIR/src/lib
	echo 'Bandit - PASSED'
	. /WorkRW/DataDumps/$(whoami)/etc/config.conf
	echo 'Checking Unit Test...'
	python3 $SP_DIR/src/lib/__rlab_common-unittest.py
	if [ $? -ne 0 ]; then exit 1; fi
	echo 'Unit Tests - PASSED'
	echo 'Checking Unit Test Coverage...'
	cd $SP_DIR/src/lib
	coverage erase
	coverage run --branch --include=rlab_common.py,rlab_common_numpy_pandas.py $SP_DIR/src/lib/__rlab_common-unittest.py
	if [ $? -ne 0 ]; then exit 1; fi
	coverage report --fail-under=37 -m $SP_DIR/src/lib/rlab_common*.py
	if [ $? -ne 0 ]; then exit 1; fi
	echo 'Coverage - PASSED'
	echo 'Reinstalling Into Current Session User '$(whoami)'...'
	cd $(realpath $SP_DIR)
	pip3 wheel --wheel-dir=wheel --no-index --find-links=wheel $(realpath $SP_DIR/src)
	pip3 install --user --no-index --find-links=wheel rlab_common
	echo 'Reinstalling '$SP_DIR'...DONE'
}
if [ ! -d $LOGD ]; then mkdir -p $LOGD; fi
pip3 uninstall rlab_common
f_reinstall 1>> $LOGF 2>> $LOGF
