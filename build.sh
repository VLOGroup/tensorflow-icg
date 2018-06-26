#!/bin/bash
# Check for gcc version according to
# https://unix.stackexchange.com/questions/285924/how-to-compare-a-programs-version-in-a-shell-script
currentgccver="$($GCC_HOST_COMPILER_PATH -dumpversion)"
requiredgccver="5.0.0"

OPTFLAG=""
 if [ "$(printf "$requiredgccver\n$currentgccver" | sort -V | head -n1)" == "$currentgccver" ] && [ "$currentgccver" != "$requiredgccver" ]; then
        echo "GCC version less than " $requiredgccver ". No additional flags required."
 else
        echo "GCC version greater than" $requiredgccver ". Build with additional flag --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0"
        OPTFLAG="--cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0"
 fi


RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# try to build tensorflow
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package $OPTFLAG
rc=$?
# check if everything is alright
if [[ $rc != 0 ]]; then
	# an error occured
	printf "${RED} BUILD ERROR! Skipping package generation.${NC}\n"
else
	# build the wheel package
	TARGET_PIP_DIR='/tmp/tensorflow_pkg'
	bazel-bin/tensorflow/tools/pip_package/build_pip_package ${TARGET_PIP_DIR}
	#sudo pip install /tmp/tensorflow_pkg/tensorflow-1.2.1-py2-none-any.whl --upgrade

	# Get generated package name
	PKG_NAME=$(ls -t ${TARGET_PIP_DIR} | head -1)

	# print install information
	printf "${RED}Remember to install:\n\n"
	printf "\t pip install ${TARGET_PIP_DIR}/${PKG_NAME} --upgrade\n"
	printf "${NC}\n"
fi

exit $?
