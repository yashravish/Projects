# myscript.sh - Batch Mode Test Script for mysh

# Test built-in pwd command
pwd

# Test change directory and error handling for cd
cd ..
pwd
cd non_existent_directory

# Test which command (assuming "ls" is an external command)
which ls

# Test wildcard expansion (adjust the pattern according to your current directory files)
echo "Testing wildcard expansion:"
ls *.c

# Test output redirection: write output to a file
echo "This is a test for output redirection." > redir_output.txt

# Test input redirection: display contents of the redir_output.txt file
cat < redir_output.txt

# Test pipeline: pipe the output of cat into grep (assuming grep is available)
cat redir_output.txt | grep "test"

# Test conditional execution:
# The following will not print if the preceding command (false) fails.
false
and echo "This should not print because the previous command failed."
or echo "This prints because the previous command failed."

# Terminate the shell in batch mode
exit