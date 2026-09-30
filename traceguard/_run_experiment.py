import sys, os
sys.path.insert(0, r"c:\Users\DESHNA\TraceGuard\traceguard")
os.chdir(r"c:\Users\DESHNA\TraceGuard\traceguard")

# Run the full experiment  
exec(open(r"c:\Users\DESHNA\TraceGuard\traceguard\experiments\run_lstm.py").read())
main()
