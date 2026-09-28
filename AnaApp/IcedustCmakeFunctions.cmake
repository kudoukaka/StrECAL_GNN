
# IcedustOwnProjectFunctions
# --------------------------
#
# DOCUMENTATION HERE -- TODO
#
# manager COMET Software Group
# author andreas.jansen@tu-dresden.de




# ICEDUST_FIND_INSTALL_AND_ROOT
# -----------------------------
#
# DOCUMENTATION HERE -- TODO
#

function(ICEDUST_FIND_INSTALL_AND_ROOT)
  if((NOT ICEDUST_ROOT) OR (NOT EXISTS ${ICEDUST_ROOT}))
    # Not set yet... try to fetch it from environment
    set(ICEDUST_ROOT $ENV{ICEDUSTROOT})
    if((NOT ICEDUST_ROOT) OR (NOT EXISTS ${ICEDUST_ROOT}))
      message(FATAL_ERROR "Could not find ICEDUST sources! Please specify ICEDUST_ROOT by:
      a) setting ICEDUSTROOT environment variable 
      b) providing it with an option: -DICEDUST_ROOT=/path/to/ICEDUST_ROOT/
      c) editing CMakeLists.txt")
    endif()
  endif()

  if((NOT ICEDUST_INSTALL_PATH) OR (NOT EXISTS ${ICEDUST_INSTALL_PATH}))
    # Not set yet... try to fetch it from environment
    set(ICEDUST_INSTALL_PATH $ENV{ICEDUSTINSTALLPATH})
    if((NOT ICEDUST_INSTALL_PATH) OR (NOT EXISTS ${ICEDUST_INSTALL_PATH}))
      # Try to set the path to standard location: ${ICEDUST_ROOT}/ICEDUST_install
      if(EXISTS ${ICEDUST_ROOT}/ICEDUST_install)
        set(ICEDUST_INSTALL_PATH ${ICEDUST_ROOT}/ICEDUST_install)
      else()
        message(FATAL_ERROR "Could not find ICEDUST libraries! Please specify ICEDUST_INSTALL_PATH by:
        a) setting ICEDUSTINSTALLPATH environment variable 
        b) providing it with an option: -DICEDUST_INSTALL_PATH=/path/to/ICEDUST_INSTALL_PATH/
        c) editing CMakeLists.txt")
      endif()
    endif()
  endif()

  set(ICEDUST_INSTALL_PATH ${ICEDUST_INSTALL_PATH} PARENT_SCOPE)
  set(ICEDUST_ROOT ${ICEDUST_ROOT} PARENT_SCOPE)
endfunction()




# ICEDUST_CONFIGURE_OUTPUT
# ------------------------
#
# DOCUMENTATION HERE -- TODO
#

function(ICEDUST_CONFIGURE_PROJECT)
  # --- Default CMAKE_INSTALL_PREFIX ---
  if(CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")
      # -- TODO -- default CMAKE_INSTALL_PREFIX under linux is "/usr/local"
      # so if no other PATH is provided we set our default prefix path here
      # this of course disables "/usr/local" as a valid option which is fine for now(?) -- FIXME
      get_filename_component(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR} ABSOLUTE)
  endif()
  set(CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} PARENT_SCOPE)

  # --- RPATH handling for shared library linking ---
  ## for more infos on this topic see: https://cmake.org/Wiki/CMake_RPATH_handling 
  # use, i.e. don't skip the full RPATH for the build tree
  SET(CMAKE_SKIP_BUILD_RPATH FALSE PARENT_SCOPE)
  # when building, don't use the install RPATH already (but later on when installing)
  SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE PARENT_SCOPE) 
  # add the automatically determined parts of the RPATH
  # which point to directories outside the build tree to the install RPATH
  SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE PARENT_SCOPE)
endfunction()




# ICEDUST_CONFIGURE_TARGETS
# -------------------------
#
# DOCUMENTATION HERE -- TODO
#

function(ICEDUST_CONFIGURE_TARGETS)
  set(ICEDUST_TARGETS "")

  foreach(packageName ${ICEDUST_DEPENDENT})
    # Library UPPER name
    string(TOUPPER ${packageName} packageName_UPPER)

    # Find the library of the package
    set(LIB_VAR "${packageName_UPPER}_LIBRARIES")
    find_library(${LIB_VAR} 
                 ${packageName} 
                 PATHS ${ICEDUST_INSTALL_PATH} NO_DEFAULT_PATH 
                 PATH_SUFFIXES lib64 lib ${packageName}/lib ${packageName}/lib64)
    set(temp1 "${${LIB_VAR}}")
    if(temp1)
      message(STATUS "Found ${packageName} library: ${temp1}")
    else()
      message(WARNING "Could not find ${packageName} library...")
    endif()

    # Find the include directory
    set(INC_DIR_VAR "${packageName_UPPER}_INCLUDE_DIRS")
    set(SEARCH_DIRS
      ${ICEDUST_INSTALL_PATH}/${packageName}/src
      ${ICEDUST_INSTALL_PATH}/${packageName}/include
      ${ICEDUST_ROOT}/ICEDUST_packages/${packageName}/src
      ${ICEDUST_ROOT}/ICEDUST_packages/${packageName}/include
      )
    foreach(search ${SEARCH_DIRS})
      if(EXISTS ${search})
        set(${INC_DIR_VAR} ${search})
        break()
      endif()
    endforeach()
    set(temp2 "${${INC_DIR_VAR}}")
    if(temp2)
      message(STATUS "Found ${packageName} includes: ${temp2}")
    else()
      message(WARNING "Could not find ${packageName} includes...")
    endif()

    # Define target to link against
    if((NOT temp1) AND (NOT temp2))
      message(WARNING "Could not find includes nor library of ${packageName}, Skipped!")
      continue()
    endif()
    set(CURRENT_TARGET "${packageName_UPPER}::${packageName_UPPER}")
    list(APPEND ICEDUST_TARGETS ${CURRENT_TARGET})
    add_library(${CURRENT_TARGET} UNKNOWN IMPORTED)
    set_property(TARGET ${CURRENT_TARGET} APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
    set_target_properties(${CURRENT_TARGET} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${temp2}")
    set_target_properties(${CURRENT_TARGET} PROPERTIES IMPORTED_LOCATION_RELEASE "${temp1}")
  endforeach()

  set(ICEDUST_TARGETS ${ICEDUST_TARGETS} PARENT_SCOPE)
endfunction()




# ICEDUST_CONFIGURE_OUTPUT
# ------------------------
#
# Macro to set up the target properties for make and install step
# and set up RPATH for our targets
#
# Usage
# ^^^^^
#
#---ICEDUST_CONFIGURE_OUTPUT(TARGETS target1 [target2 ...])
# 
# TARGETS specifies all target which shall be configured
#

function(ICEDUST_CONFIGURE_OUTPUT)
  CMAKE_PARSE_ARGUMENTS(ARG "" "" "TARGETS" ${ARGN})

  set_target_properties(${ARG_TARGETS}
    PROPERTIES
    # Standard-Output-Directories for make: ../build_directory/packageXYZ/
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
  )

  # Standard-Output-Directories for "make install"
  # "CMAKE_INSTALL_PREFIX" is prepended to all directories!
  install(TARGETS ${ARG_TARGETS}
    RUNTIME DESTINATION ${PROJECT_NAME}/bin
    LIBRARY DESTINATION ${PROJECT_NAME}/lib
    ARCHIVE DESTINATION ${PROJECT_NAME}/lib
  )
endfunction()
