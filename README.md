Installation and Compilation Instructions
This document will guide you through the process of setting up your environment and compiling the Predictive Software into an executable file.
Prerequisites
You will need to install Python and several packages. Don't worry if you're not familiar with Python - the instructions below will walk you through every step.
Step 1: Install Python
1.	Download Python from the official website: https://www.python.org/downloads/
o	Select the "Download Python" button (choose the latest version, like Python 3.10)
o	Make sure to check the box that says "Add Python to PATH" during installation
2.	Verify Python is installed by opening a Command Prompt (Windows) or Terminal (Mac/Linux):
python --version
You should see a message showing the Python version.
Step 2: Install Required Packages
1.	Open a Command Prompt (Windows) or Terminal (Mac/Linux)
2.	Install the required packages by typing the following commands (press Enter after each line):
pip install numpy
pip install pandas
pip install pykalman
pip install filterpy
pip install matplotlib
pip install pyinstaller
Compiling the Application
Step 1: Save the Python Script
1.	Copy the entire Python code I provided
2.	Open a text editor (like Notepad, TextEdit, or VS Code)
3.	Paste the code into the editor
4.	Save the file as predictive_software.py (make sure it has the .py extension)
Step 2: Create the Executable
1.	Open a Command Prompt (Windows) or Terminal (Mac/Linux)
2.	Navigate to the folder where you saved the Python file
o	For example, if you saved it on your Desktop: 
	Windows: cd C:\Users\YourUsername\Desktop
	Mac/Linux: cd ~/Desktop
3.	Run PyInstaller to create the executable:
pyinstaller --onefile --windowed predictive_software.py
This will create a standalone executable file.
4.	The compilation process will take several minutes. When it's finished, you'll find the executable in the dist folder that was created in the same directory.
Step 3: Run the Application
1.	Navigate to the dist folder
2.	You'll find a file named predictive_software.exe (Windows) or predictive_software (Mac/Linux)
3.	Double-click this file to run the application
Troubleshooting
If you encounter any issues during installation or compilation:
Package Installation Issues
If you get errors when installing packages, try running pip with administrator privileges:
•	Windows: Right-click on Command Prompt and select "Run as administrator"
•	Mac/Linux: Use sudo pip install [package_name]
Missing DLL or Library Files
If the executable gives errors about missing DLLs or libraries:
1.	Try creating the executable with all dependencies included: 
pyinstaller --onefile --windowed --hidden-import=pykalman --hidden-import=filterpy.kalman --hidden-import=matplotlib predictive_software.py
Executable Not Starting
If the executable doesn't start:
1.	Open a Command Prompt or Terminal
2.	Navigate to the dist directory
3.	Run the executable from the command line to see any error messages: 
o	Windows: predictive_software.exe
o	Mac/Linux: ./predictive_software
Additional Notes
•	The first time you run the executable, it might take a little longer to start
•	Windows might show a security warning - click "More info" and then "Run anyway"
•	On Mac, you might need to go to System Preferences > Security & Privacy and allow the app to run

