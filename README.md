# capstone-ai-production
Capstone project for ML

# Create Docker Image
1. Create a directory "capstone-proj-docker" on your computer at any convient location. e.g Users/yourid/Documents/capstone-proj-docker
2. Copy capstoneproject.py, requirements.txt, Dockerfile in capstone-proj-docker
3. Open terminal and navigate to capstone-proj...e.g cd Users/yourid/Documents/capstone-proj-docker
4. docker build -t capstone-proj-docker .
5. docker run -it capstone-proj-docker capstoneproject.py
6. All docker creation files can be found in capstone-proj-docker

# Run Unit Test Cases
1. Get all the files from capstone-proj-testcase
2. If running the test cases from Jupyter Notebook, then open testcapstoneproject.ipynb in jupyter notebook and run.
3. If running from command line, issue the following command from command line..
           $ python -m unittest testcapstoneproject
