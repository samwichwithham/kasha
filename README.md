Kasha Video Classifier (v0.1.3)

Kasha is a desktop application that scans folders of video footage to automatically identify and tag interview clips and b-roll.

1. First-Time Setup (Do This Only Once)

Before running the project, you need to install a few required tools on your computer.

1.1 Install Homebrew

Homebrew is a package manager for macOS. Open the Terminal app and paste this command:

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

1.2 Install Required Software

Use Homebrew to install Git, Python, and the necessary GUI toolkit.

brew install git
brew install python@3.11
brew install python-tk@3.11

1.3 Install PDM (Python Dependency Manager)

This command installs the tool that manages the project's libraries. You may need to close and reopen your terminal after this step.

curl -sSL https://pdm-project.org/install-pdm.py | python3 -

2. Download and Set Up the Kasha Project

This is how you get the project on your computer and install its specific libraries.

2.1 Choose a Location and Download the Project

First, navigate in your terminal to a place where you want to store your code (like your Desktop or a folder like ~/Documents/Code). Then, run this command to download the project. It will create a new folder named kasha in your current location.

git clone https://github.com/samwichwithham/kasha.git

2.2 Enter the Project Folder

Now, move into the newly created project folder.

cd kasha

2.3 Set the Python Version for the Project

pdm use python3.11

2.4 Install All Project Libraries

This command reads the project files and automatically installs everything you need.

pdm install

2.5 Download the AI Language Model

pdm run python -m spacy download en_core_web_sm

3. How to Run the Application

After the one-time setup is complete, you only need to do this every time you want to run the app.

    Navigate to the project folder in your terminal (e.g., cd kasha).

    Run this command:

    pdm run python run_gui.py
