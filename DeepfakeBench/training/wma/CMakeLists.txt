cmake_minimum_required(VERSION 3.20)

# Python Backend Service (Python) - gRPC Server
project(python_backend_service VERSION 1.0.0 LANGUAGES NONE)

# Find Python executable
find_program(PYTHON_EXECUTABLE 
    NAMES python3 python
    REQUIRED
)

if(NOT PYTHON_EXECUTABLE)
    message(FATAL_ERROR "Python executable not found. Required for python_backend service.")
endif()

# Verify Python version (3.8+)
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    OUTPUT_VARIABLE PYTHON_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(PYTHON_VERSION VERSION_LESS "3.8")
    message(FATAL_ERROR "Python 3.8 or higher required. Found: ${PYTHON_VERSION}")
endif()

message(STATUS "Found Python ${PYTHON_VERSION}: ${PYTHON_EXECUTABLE}")

# Check if PyInstaller is available
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import PyInstaller; print('PyInstaller available')"
    RESULT_VARIABLE PYINSTALLER_CHECK
    OUTPUT_QUIET ERROR_QUIET
)

if(NOT PYINSTALLER_CHECK EQUAL 0)
    message(STATUS "PyInstaller not found. Installing PyInstaller...")
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -m pip install pyinstaller
        RESULT_VARIABLE PIP_INSTALL_RESULT
    )
    
    if(NOT PIP_INSTALL_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to install PyInstaller. Please install manually: pip install pyinstaller")
    endif()
endif()

# Create executable using PyInstaller (following same pattern as other services)
set(PYTHON_BACKEND_MAIN "${CMAKE_CURRENT_SOURCE_DIR}/start_backend.py")
set(PYTHON_BACKEND_RELEASE_DIR "${CMAKE_CURRENT_BINARY_DIR}/Release")
set(PYTHON_BACKEND_EXE "${PYTHON_BACKEND_RELEASE_DIR}/python_backend.exe")

# Ensure Release directory exists
file(MAKE_DIRECTORY ${PYTHON_BACKEND_RELEASE_DIR})

# Custom command to build executable with PyInstaller
add_custom_command(
    OUTPUT ${PYTHON_BACKEND_EXE}
    COMMAND ${PYTHON_EXECUTABLE} -m PyInstaller
        --onefile
        --name python_backend
        --distpath "${PYTHON_BACKEND_RELEASE_DIR}"
        --workpath "${CMAKE_CURRENT_BINARY_DIR}/pyinstaller_build"
        --specpath "${CMAKE_CURRENT_BINARY_DIR}/pyinstaller_build"
        --clean
        --noupx
        --console
        "${PYTHON_BACKEND_MAIN}"
    DEPENDS 
        ${PYTHON_BACKEND_MAIN}
        server.py
        wma_streaming_pb2.py
        wma_streaming_pb2_grpc.py
        simple_test.py
        __init__.py
        storage/__init__.py
        storage/data_writer.py
        utils/__init__.py
        utils/banner_simulator.py
        requirements.txt
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Building python_backend.exe with PyInstaller in Release directory"
)

# Add custom target
add_custom_target(python_backend ALL
    DEPENDS ${PYTHON_BACKEND_EXE}
    SOURCES
        start_backend.py
        server.py
        wma_streaming_pb2.py
        wma_streaming_pb2_grpc.py
        simple_test.py
        __init__.py
        storage/__init__.py
        storage/data_writer.py
        utils/__init__.py
        utils/banner_simulator.py
        requirements.txt
        protos/wma_streaming.proto
)

# Install Python source files (for debugging/development)
install(FILES
    start_backend.py
    server.py
    wma_streaming_pb2.py
    wma_streaming_pb2_grpc.py
    simple_test.py
    __init__.py
    requirements.txt
    DESTINATION bin/python_backend
)

install(FILES
    storage/__init__.py
    storage/data_writer.py
    DESTINATION bin/python_backend/storage
)

install(FILES
    utils/__init__.py
    utils/banner_simulator.py
    DESTINATION bin/python_backend/utils
)

install(FILES
    protos/wma_streaming.proto
    DESTINATION bin/python_backend/protos
)

# Install PyInstaller-generated executable
install(PROGRAMS ${PYTHON_BACKEND_EXE}
    DESTINATION bin
)

# All required modules are part of Python standard library, no dependency check needed

# Set target properties for IDE integration
set_target_properties(python_backend PROPERTIES
    FOLDER "WMA Services"
    VS_DEBUGGER_COMMAND ${PYTHON_EXECUTABLE}
    VS_DEBUGGER_COMMAND_ARGUMENTS "${PYTHON_BACKEND_MAIN}"
    VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Custom target for running service in development
add_custom_target(run_python_backend
    COMMAND ${PYTHON_EXECUTABLE} ${PYTHON_BACKEND_MAIN}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running python_backend service in development mode"
    USES_TERMINAL
)