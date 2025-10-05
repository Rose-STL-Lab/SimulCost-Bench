"""Test lazy import to verify taichi is not loaded unless ns_transient_2d is used."""
import sys
import time

print("=" * 60)
print("Test 1: Import tool_call without using ns_transient_2d")
print("=" * 60)

start = time.time()
import tool_call
end = time.time()

# Check if taichi was loaded
taichi_loaded = 'taichi' in sys.modules
print(f"Import time: {end - start:.4f} seconds")
print(f"Taichi loaded: {taichi_loaded}")

if taichi_loaded:
    print("❌ FAIL: Taichi was loaded on initial import")
else:
    print("✅ PASS: Taichi not loaded on initial import")

print("\n" + "=" * 60)
print("Test 2: Access a non-ns_transient_2d function")
print("=" * 60)

# Try using a heat_1d function
try:
    func = tool_call.heat_1d_check_converge_cfl
    taichi_loaded_after_heat = 'taichi' in sys.modules
    print(f"Accessed heat_1d_check_converge_cfl successfully")
    print(f"Taichi loaded: {taichi_loaded_after_heat}")

    if taichi_loaded_after_heat:
        print("❌ FAIL: Taichi was loaded when accessing heat_1d function")
    else:
        print("✅ PASS: Taichi still not loaded after accessing heat_1d")
except Exception as e:
    print(f"❌ Error accessing function: {e}")

print("\n" + "=" * 60)
print("Test 3: Access ns_transient_2d function (should load taichi)")
print("=" * 60)

# Now access ns_transient_2d function
try:
    start = time.time()
    func = tool_call.ns_transient_2d_check_converge_resolution
    end = time.time()

    taichi_loaded_final = 'taichi' in sys.modules
    print(f"Lazy import time: {end - start:.4f} seconds")
    print(f"Accessed ns_transient_2d_check_converge_resolution successfully")
    print(f"Taichi loaded: {taichi_loaded_final}")

    if taichi_loaded_final:
        print("✅ PASS: Taichi loaded when needed")
    else:
        print("❌ FAIL: Taichi should be loaded now")
except Exception as e:
    print(f"❌ Error accessing ns_transient_2d function: {e}")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("The lazy import mechanism allows you to:")
print("- Import tool_call quickly without loading taichi")
print("- Use all other datasets without taichi overhead")
print("- Load taichi only when ns_transient_2d is actually used")
