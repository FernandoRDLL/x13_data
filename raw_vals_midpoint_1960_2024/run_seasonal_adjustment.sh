#!/bin/bash

# Loop over all .spc files in the current directory
for file in *.spc
do
  # Remove the '.spc' extension from the filename
  filename_without_extension="${file%.spc}"

  # Execute the command on the filename without the extension and append the output to general_output.txt
  ./x13as_html "$filename_without_extension" >> general_output.txt
done

