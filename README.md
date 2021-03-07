# Capstone-AI-Production
Capstone project for ML

Project Highlights
1. Select best model from multiple models and make predictions.
2. Test driven development with Unit Tests.
3. Effetive logging and unit tests for logging.
4. Run all tests as a single script.
5. Visualization and comparison using visualization.
6. Dockerize the projct to be deployed on a container platform.
7. Documentation to run the project using popular Jupyter Notebook or from command line.

# Run project
To run the project using Jupyter Notebook, open capstoneproject.ipynb in jupyter notebook and run the same.

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
