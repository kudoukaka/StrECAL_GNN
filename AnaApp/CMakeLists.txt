
# Example code to create own analysis code
#
# manager COMET Software Group
# author comet_sw@ml.post.kek.jp



# --- Min CMake version ---
cmake_minimum_required (VERSION 3.0)
# --- CMake policys ---
cmake_policy(SET CMP0048 NEW)
# --- Project name, version ---
project(TestCode LANGUAGES CXX VERSION 1.0.0)
# --- Include our cmake-macros 
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/build ${CMAKE_MODULE_PATH})
include(IcedustCmakeFunctions) 


# Obtain locations: ICEDUST_ROOT | ICEDUST_INSTALL_PATH
# -----------------------------------------------------
# Optional: set up locations here, not recommended! 
#   Better set via option: -DICEDUST_ROOT=/path/   
#   or environment (ICEDUST_ROOT, ICEDUST_INSTALL_PATH)!
#set(ICEDUST_ROOT "/path/here")
#set(ICEDUST_INSTALL_PATH "/path/here")
ICEDUST_FIND_INSTALL_AND_ROOT()


# Set RPATH options and CMAKE_PREFIX_PATH default
# -----------------------------------------------
ICEDUST_CONFIGURE_PROJECT()


# Specify dependency on ICEDUST packages
# --------------------------------------
set(ICEDUST_DEPENDENT oaEvent oaStrECALUtil oaChanInfo oaGeomInfo oaRuntimeParameters EventLoop AnalysisBase SimDetectorResponse) 


# Creating Source- / Header-Lists
# -------------------------------
# >>>> Add your files to the list(s) if you want them to be compiled <<<<
set(SourceFiles
  	${CMAKE_CURRENT_SOURCE_DIR}/DetRespAna.cxx 
  	${CMAKE_CURRENT_SOURCE_DIR}/AddStrEcalRandomNoise.cxx
)

set(HeaderFiles
)


# Create importet targets (no changes required here, change ICEDUST_DEPENDENT!)
#------------------------------------------------------------------------------
ICEDUST_CONFIGURE_TARGETS()


# Load ROOT library
#------------------
if((NOT ROOT_ROOT) OR (NOT EXISTS ${ROOT_ROOT}))
  # Optional: modify this path if other ROOT version shall be used, not recommended!
  #   Better set via option: -DROOT_ROOT=/path/to/other/ROOT
  set(ROOT_ROOT ${ICEDUST_ROOT}/ICEDUST_externals_install/root_v6.18.00/)
endif()
list(APPEND CMAKE_PREFIX_PATH ${ROOT_ROOT})
find_package(ROOT REQUIRED COMPONENTS Core Tree Geom Physics EG Eve Ged RGL
    Gui Matrix RIO Hist XMLParser Net MathCore MathMore Gpad Graf Graf3d Spectrum Minuit) 
include(${ROOT_USE_FILE})


# Optional: Create ROOT-Dictionaries (use cmake/Modules/IcedustCreateROOTDict macro) if LinkDefs exist inside /src/
#------------------------------------------------------------------------------------------------------------------
# list(APPEND CMAKE_MODULE_PATH ${ICEDUST_ROOT}/ICEDUST_packages/cmake/Modules)
# include(IcedustCreateROOTDict) 
# ICEDUST_GENERATE_DICTIONARY("${CMAKE_CURRENT_SOURCE_DIR}/src/ExampleClassName_Dict" 
#   HEADER ${CMAKE_CURRENT_SOURCE_DIR}/src/ExampleClassName.hxx
#   DEPENDENCIES ${ICEDUST_DEPENDENT}
#   LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/src/ExampleClassName_Linkdef.h
#   OPTIONS -p)
# list(APPEND ROOT_DICTS ${CMAKE_CURRENT_SOURCE_DIR}/src/ExampleClassName_Dict.cxx) 


# build the executables
#----------------------
# TestProject
#add_executable(TestProject ${SourceFiles} ${HeaderFiles} ${ROOT_DICTS})
#target_link_libraries(TestProject PRIVATE ${ICEDUST_TARGETS} ${ROOT_LIBRARIES})
#target_include_directories(TestProject PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(DetRespAna ${CMAKE_CURRENT_SOURCE_DIR}/DetRespAna.cxx)
target_link_libraries(DetRespAna PRIVATE ${ICEDUST_TARGETS} ${ROOT_LIBRARIES})
target_include_directories(DetRespAna PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(AddStrEcalRandomNoise ${CMAKE_CURRENT_SOURCE_DIR}/AddStrEcalRandomNoise.cxx)
target_link_libraries(AddStrEcalRandomNoise PRIVATE ${ICEDUST_TARGETS} ${ROOT_LIBRARIES})
target_include_directories(AddStrEcalRandomNoise PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Example for setting preprocessor directives
# target_compile_definitions(TestProject PUBLIC definition)

# Example to include additional directories which contains needed Header/Sources
# target_include_directories(TestProject PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/example)

# Example to add a test to this application to be run with "make test"
# add_test(NAME TestProject COMMAND TestProject)



# specify output-directories for "make" and "make install"
#---------------------------------------------------------
ICEDUST_CONFIGURE_OUTPUT(TARGETS DetRespAna AddStrEcalRandomNoise)
