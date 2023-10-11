---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Overall project checklist

The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

## Week 1
- [x] Create a git repository

- [x] Make sure that all team members have write access to the github repository

- [ ] Create a dedicated environment for you project to keep track of your packages
    -> add the conda exported file to github @maya

- [x] Create the initial file structure using cookiecutter

- [x] Fill out the make_dataset.py file such that it downloads whatever data you need and

- [x] Add a model file and a training script and get that running

- [x] Remember to fill out the requirements.txt file with whatever dependencies that you are using

- [x] Remember to comply with good coding practices (pep8) while doing the project

- [x] Do a bit of code typing and remember to document essential parts of your code

- [x] Setup version control for your data or part of your data

    - was there in dvc originally, is now done in cloud storage with versioning

- [x] Construct one or multiple docker files for your code

- [x] Build the docker files locally and make sure they work as intended

- [x] Write one or multiple configurations files for your experiments

- [x] Used Hydra to load the configurations and manage your hyperparameters

- [ ] When you have something that works somewhat, remember at some point to to some profiling and see if you can optimize your code

    - profiling & optimization once model is optimized?

- [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally, consider running a hyperparameter optimization sweep.

- [x] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

## Week 2

- [x] Write unit tests related to the data part of your code

- [x] Write unit tests related to model construction and or model training

- [x] Calculate the coverage. = 91% :)

- [x] Get some continuous integration running on the github repository

- [x] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup

    - link this with your data version control setup: not useful in our case

- [x] Create a trigger workflow for automatically building your docker images

    - trigger on push to main

- [x] Get your model training in GCP using either the Engine or Vertex AI

    - Training and tuning is done in compute engine

- [ ] Create a FastAPI application that can do inference using your model

- [ ] If applicable, consider  the model locally using torchserve

    - not useful

- [ ] Deploy your model in GCP using either Functions or Run as the backend

    - check whether useful, because model needs to loaded every time

    - have everything in engine? Or use maybe vertex

## Week 3

- [ ] Check how robust your model is towards data drifting

    - part of fridays video, deploy db??

- [ ] Setup monitoring for the system telemetry of your deployed model

- [ ] Setup monitoring for the performance of your deployed model

- [ ] If applicable, play around with distributed data loading
      - Not applicable, data is less than 100MB when zipped

- [ ] If applicable, play around with distributed model training
      - Not applicable, models train in a few seconds on CPU

- [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

***

##   Addtional

- [ ] Revisit your initial project description. Did the project turn out as you wanted?

- [ ] Make sure all group members have a understanding about all parts of the project

- [x] Uploaded all your code to github


## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

Maya Köhnen and Jan Anders

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

Not posting this here for privacy. Please see our submission on Moodle.

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We chose to work with a combination of tools and frameworks that best suited our ML Ops project. Our primary framework for machine learning was PyTorch Lightning, which provided us with a high-level interface for PyTorch, streamlining the development of our machine learning models. This choice greatly expedited the model development phase.

In terms of infrastructure, Google Compute Engine served as our cloud computing platform, facilitating scalable and efficient model training. Additionally, we used SMAC for hyperparameter tuning, allowing us to optimize our models effectively.

For continuous integration and continuous deployment (CI/CD), we relied on GitHub Actions, which seamlessly integrated with our GitHub repository. This CI/CD pipeline automates testing, building Docker images, and deploying our model artifacts to DockerHub.


## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development enviroment, one would have to run the following commands*
>
> Answer:

We managed dependencies in our project using a combination of tools and approaches to ensure a streamlined environment setup process for new team members. First, we utilized pip and conda for Python package management. The list of project-specific Python dependencies was stored in a requirements.txt file as well es in an conda environment export, making it easy to maintain and reproduce the environment needed for development. We install different dependencies from respective requirement files for the different docker containers setup for different tasks, e.g. training or inference.

To ensure a consistent Python version across the team, we specified Python 3.11 as the required version. Additionally, Docker was used for containerization, and DockerHub hosted our Docker images. This allowed us to encapsulate the entire project environment, including dependencies and configurations, within Docker containers.

For local development, we provide a Conda environment export file that contains all necessary dependencies. A new team member could set up the development environment by following these steps:

1. Install Docker if not already installed.
2. Clone the project repository from GitHub.
3. Navigate to the project directory.
4. Create a virtual environment using Conda with the exported environment file.
   ```
   conda env create -f mlops_23_env.yaml
   ```
5. Activate the Conda environment.
   ```
   conda activate mlops_23
   ```
6. Build and run Docker containers with the project image.


### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

While we did not use every folder provided by the cookiecutter template, we found it to be a helpful starting point for structuring our project. Here's an overview of how we organized our code:

1. **Data (Used)**: The "data" folder played a crucial role in our project. It contained raw and processed datasets as well as the data colleceted from inference. This folder was essential for data ingestion and preparation.

2. **Src (Used)**: The "src" folder housed our source code, including machine learning model scripts, data processing scripts, and FastAPI implementation. It formed the core of our project's functionality.

3. **Models (Used)**: The "models" folder stored our trained machine learning models, i.e. model checkpoints.

4. **Report (Used)**: We utilized the "report" folder to store Pareto plots generated by SMAC. This helped us analyze and visualize hyperparameter tuning results effectively.

5. **Tests (Used)**: The "tests" folder contained unit tests and test data. Ensuring the reliability and correctness of our code was vital, and this folder facilitated automated testing.

6. **Notebooks (Not Used)**: We did not use the "notebooks" folder for our project. Our development workflow primarily revolved around scripts rather than Jupyter notebooks.

7. **References (Not Used)**: The "references" folder was not utilized in our project as it did not align with our specific requirements.

8. **Outputs (Not Used)**: Similarly, the "outputs" folder was not employed in our project, as we had different mechanisms for storing and managing outputs.

9. **.github (Used)**: We added a ".github" folder to leverage GitHub Actions for continuous integration and continuous deployment (CI/CD). This folder contained workflows for automating various project tasks.

10. **.dvc (Used)**: We also included a ".dvc" folder to integrate Data Version Control (DVC) into our project. However data versioning was done in the cloud once integrated.

We tailored the cookiecutter template to our project's specific needs, utilizing the folders that were most relevant to our ML Ops workflow while omitting those that did not align with our development approach.

### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

In our project, we emphasized code quality by implementing separation of concerns w.r.t. python modules, classes and function, to keep code parts as small and separated as possible. Including type hints and comprehensive docstrings where applicable additionally imporoved code quality as well as the unit tests. They did not only enforced a modularized implementation they also allowed us to automatically check that code changes, which improves quality w.r.t. errors. The tests were run manualyy but also on every push to a pull request aiming to be merged into the main branch.

To maintain consistent code formatting, we relied on tools like flake8, black, and isort. We used pre-commit hooks to ensure the code conforms to these requirements at all time. We increased the max-line-length to a higher value, since the line-too-long messages result from inconsistent expectations in flake8 and black.

Although making sure all linters don't report any errors can be bothersome, especially when in a hurry to push a commit, we think these practices are vital in larger projects as they improve code maintainability, foster effective collaboration among team members, enhance code readability, reduce the likelihood of errors, and ensure uniformity in coding standards throughout the project, ultimately leading to a more robust and manageable codebase.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

We implemented 18 tests, that focus on the core parts of a machine learning application. This are namely the data, the model, it's training and inference.
For data tests concern shape checks and preprocessing, the model is also tested w.r.t. shapes of in- and output. For training we ensure correctness of basic functionalities like
model creation, hyperparameters and logging. Inference is one of the most vital but also error prone part of our application as it is exposed to day to day usage. Therefor it has the largest share of tests, that ensure basic availability of the application as well as correct responses to different user inputs.

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

Our code coverage stands at 91%, indicating that our unit tests cover a substantial portion of our codebase. While a high code coverage percentage is an encouraging sign, it doesn't guarantee that the code is entirely error-free. Code coverage primarily measures which parts of the code have been executed during testing, but it doesn't assess the quality of the tests themselves.

Even with near 100% code coverage, we cannot ensure error free code because it's possible that certain edge cases or complex scenarios may not be adequately tested. Additionally, the quality of the tests matters; poorly designed or insufficient tests may not catch all potential issues.

Therefore, while high code coverage is essential, it should be complemented with thorough testing strategies, code reviews, and other quality assurance measures to enhance confidence in the code's reliability and correctness.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

Our workflow extensively utilized branches and pull requests (PRs) to manage code changes:

1. **Branch Development**: We (mostly) followed a branch-per-feature approach, where each new feature or bug fix was developed on its dedicated branch. This isolated changes and minimized conflicts during development, though we did go through some merge conflicts along the way. The latter occured mostly when because parallel developments required changing the same code and was solved by rebasing on the main branch.

2. **Pull Requests**: To integrate changes into the main branch, we created PRs. These PRs served as a means for code review, ensuring that code quality was met and both of us always had an overview of what the other person was doing. PRs also included references to issue numbers, linking them to project tasks and milestones.

3. **Code Review**: PRs underwent a code review process by team members, discussion was handlend via comments and personal follow-up conversations. This review step helped identify and rectify potential issues.

4. **Automated Workflows**: We set up automated workflows within our PRs. This included running tests, ensuring code quality, and deploying developmental features for testing in the environments. In a real project, a staging enviroment would be needed for this.

Using branches and PRs improved version control in our project for feature development, facilitating collaboration among team members, and ensuring that code changes were reviewed and tested before merging. This approach contributed to a more organized and reliable development process.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We did initially use DVC (Data Version Control) for managing data in our project. However, we encountered challenges when trying to seamlessly integrate DVC for data and model version control in the cloud. The main issue was the complexity of transferring DVC-generated MD5 sums from the training environment to the testing container.

To address these challenges more effectively, we opted for an alternative approach. We leveraged Google Cloud Storage's built-in file versioning capabilities, complemented by a retention policy of 90 days. This approach allowed us to maintain version control of our data and models while eliminating the need for  MD5 sum or dvc config file transfers.

This solution not only streamlined our data management but can also ensure compliance with GDPR regulations for sensitive data handling, in cases where this would be necessary. In cases where data version control is crucial, especially in cloud-based environments with data transfer complexities, using cloud storage with retention policies can be a more straightforward and efficient approach. For experimentation and research, we would however highly reccommend using dvc, as it forsters easy reproducibility.

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

Our continuous integration (CI) setup is an integral part of our development process. Here's an overview:

1. **Tests**: We run unit tests to validate the correctness of our code. These tests cover various aspects of our application to ensure its reliability, as mentioned before. Running them in githubs actions using only the predefined python version and dependencies given in the requirements file helped ensuring that test outcomes are not influenced by the local development environment.

2. **Docker Image Building**: We build Docker images for training, inference, and monitoring. These images are automatically pushed to DockerHub, ensuring that we have the latest and most up-to-date versions available for deployment.

3. **Deployment on Compute Engine**: We automate the deployment of these Docker images onto Google Compute Engine. This step only gets executed if the building and push to dockerhub is successful. Getting this to work involved complexities related to authentication, as GitHub needs authorization to interact with the cloud. We've configured service accounts to grant Compute Engine the necessary rights for storing objects in Google Cloud Storage.

4. **Multiple Environments**: Since all code is executed in docker containers in the cloud, there are no plans to expand our CI setup to test multiple Python versions and operating systems to ensure cross-compatibility.

You can find our GitHub Actions workflows [here](https://github.com/mkoehnenlmu/mlops_project_23/tree/main/.github/workflows), which provides a detailed view of our CI setup and automation process. [Here](https://github.com/mkoehnenlmu/mlops_project_23/actions) are the executed workflows.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

In our project, we primarily focus on stable model training with the option for retraining when necessary, rather than running experiments. However, we utilize config files to manage hyperparameters and model configurations effectively. Here's an example of how we use config files:

1. **Training Config**: We have a configuration file specifying hyper-hyperparameters, such as number of SMAC configurations used during training. Hyperparams are then set dynamically by SMAC.

2. **Model Configuration**: Another file stores model-specific parameters, architecture details, and other params used in model training, but only for the best model found while tuning. Other model configs are stored by SMAC and could be exported in the future if needed.

3. **Transfer Configurations**: These config files facilitate transferring model settings from training to inference, ensuring consistency.

The latest configuration is automatically pulled by the inference docker container upon startup.

This approach helps maintain configuration consistency and simplifies the process of training and deploying models.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

As mentioned, our project does not really aim at running experiments but is designed for deploying a ML solution, so reproducing experiments is not a necessity. Nevertheless, reproducibility is important:

1. **Model and Hyperparameter Saving**: We save trained models and the hyperparameters used during training. This step allows us to precisely recreate the model's state and configuration at any given point, facilitating reproducibility.

2. **Global Seed Setting**: We set a global random seed in our configuration files. This ensures that random processes, such as data shuffling or weight initialization, are consistent across runs. By fixing the seed, we eliminate randomness and enhance reproducibility, so the only variable in training is the dataset. If it gets updated, the trained model would change.

3. **Logging and Version Control**: We maintain logs of each experiment, including key metrics, dataset information, and configuration details.

4. **Dependency Management**: We maintain a comprehensive list of dependencies and their versions. This ensures that the software environment remains consistent and reproducible across different runs on different environments.

By combining these practices, we minimize information loss, maintain experiment reproducibility, and establish a reliable foundation for our machine learning operations.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

- [A pareto-front of our training](figures/pareto.png)
Figure Description:

The figure displays a Pareto front generated during our tuning run with SMAC. It shows a trade-off relationship between two key metrics: model size and 1-F1 measure.

- Model Size: This metric represents the complexity or size of the machine learning model. It may be measured in terms of the number of parameters, memory usage, or other relevant factors. Smaller models consume fewer resources but might sacrifice performance.

- 1-F1 Measure: The 1-F1 measure is a commonly used metric in classification tasks, particularly in cases of imbalanced datasets. It combines precision and recall, providing a balanced assessment of a model's ability to correctly classify positive and negative instances. A higher 1-F1 score indicates better classification performance.

Significance:

The Pareto front is a valuable visualization of our hyperparameter tuning as it showcases the trade-offs between model size and classification performance. In this specific figure, you can see that there seem to be three different configurations with different possible achieved F1 scores where model size does not seem to matter. Thus, choosing the smallest model that achieves an F1 of 0.9 is the natural choice.

- [Maybe a loss curve from the model?]()
- [We need a third plot!!!]()


### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

Docker played a pivotal role in our experiments by containerizing our machine learning models and their dependencies, ensuring reproducibility and portability. Here's how we utilized Docker:

1. **Containerization**: As mentioned before, we encapsulated our model training, inference as well as monitoring and its associated environment within a Docker container. This container included Python dependencies, libraries, and specific configurations, ensuring that the model's behavior, i.e. training and the application behavior remained consistent across different environments.

2. **Version Control**: Build docker images are version-controlled on DockerHub, allowing us to easily access and deploy different versions of our containers.

3. **Deployment**: We deployed the docker images to the google cloud, i.e. containers set up on training, inference and monitoring images run on dedicated vm instances using Google Compute Engine. While this is not as vital for training for inference and monitoring this allows constant availability of relevant endpoints and provided a seamless and consistent environment for model serving.

To run our Docker images locally, we typically used the following command, where <image_name> can be replaced by either training, inference or monitoring:

```bash
docker build -f <image_name>.dockerfile . -t <image_name>:latest
```

```bash
docker run -it -p 8080:8080 <image_name>:latest
```

You can find an example of one of our Dockerfiles [here](https://github.com/mkoehnenlmu/mlops_project_23/blob/main/training.dockerfile), which outlines the specific dependencies and configurations used for our containerized machine learning models.

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

Debugging in a cloud-based environment posed unique challenges for our project:


1. **Local Development**: Whenever possible, we conducted local development. This environment provided access to standard debugging tools, allowing us to set breakpoints, inspect variables, and step through code more effectively. As an intermediate step to deployment the docker images where build and run locally to detect potential issues with a changing environment, where debugging could be done by locally checking the container logs or performing required actions directly within the container.

2. **GitHub Actions**: Debugging in GitHub Actions involved committing and pushing code repeatedly to uncover the issue. While effective for tracking down certain bugs, it was time-consuming and cumbersome, but no better alternatives were available.

3. **SSH Access**: For cloud-based instances, we used SSH to access virtual machines and investigate issues directly. This allowed us to examine logs, inspect configurations, and identify the root causes of problems. Since the code running in the cloud could already be tested locally, the issues here were mostly consisting of access to other cloud resources.

Regarding code profiling, we performed it to optimize code performance where necessary. While our code was well-structured, we recognized the importance of profiling to identify bottlenecks and potential areas for improvement, ensuring that our code remained efficient and scalable as we scaled up our experiments.
We are aware of possible improvements in the creation of the training dataset, but since this code is only run once, this was not a priority to improve.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We utilized the following Google Cloud Platform (GCP) services in our project:

1. **Google Cloud Compute Engine**: We deployed virtual machines on Compute Engine for both inference and training tasks, and additionally for monitoring. These VMs provided scalable computing resources for our machine learning workloads.

2. **Google Cloud Storage**: We utilized Google Cloud Storage buckets for storing various project artifacts, including datasets, trained models, and inference data and results. This offered secure, scalable and version controlled object storage for our data and assets.

Our project intentionally refrained from using additional managed GCP services to maintain cost control and improve portability. By managing our infrastructure ourselves, we ensured flexibility and avoided dependencies on specific GCP services, enhancing the project's adaptability to different platforms and environments.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

Google Cloud Compute Engine was pivotal in our project for its scalable and efficient virtual machine (VM) hosting capabilities. We leveraged Compute Engine in the following ways:

**Training VM**:
- For model training, we created VM instances with the `c2d-highcpu-8` machine type, providing ample CPU cores for parallel processing.
- We specified a custom container image from DockerHub, allowing us to encapsulate our training environment and dependencies.

**Inference VM**:
- Our inference VMs were configured with a similar setup but with fewer CPU cores, using the `c2d-highcpu-4` machine type. This was suitable for serving model predictions with lower computational demands.

**Monitoring VM**:
- The monitoring VMs were configured as the inference VM as they have similar computational demands.

All types of VMs were hosted in the `europe-west3-c` zone and were associated with specific service accounts, granting them necessary permissions for cloud platform access. Compute Engine ensured our project had the computing power required for training and serving machine learning models efficiently in a scalable and cloud-native manner.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

[Bucket Overview](figures/bucket1.png)
[Model Storage Overview](figures/bucket2.png)

### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

[Our Dockerhub Repository](figures/dockerhub.png)

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

[Our Inference Build History](figures/buildinference.png)

[Our Training Build History](figures/buildtraining.png)

As already outlined above, we wanted the project to stay as independent of the platform as possible, so we stored the built images in Dockerhub.

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We successfully deployed our machine learning model in the cloud, specifically on Google Compute Engine. We selected Google Compute Engine as our deployment environment due to its available processing power combined with flexibility in deploying any application we want. Using Cloud Run was an alternative, but we were unsure how it would handle the moderately sized inference docker container:

1. **Docker Container**: We containerized our model and its associated services within a Docker container, which allowed us to maintain consistency and encapsulate all dependencies.

2. **API Service**: We exposed our model as an API service using FastAPI, enabling external systems to send HTTP requests for predictions.

3. **Inference Endpoint**: We set up an inference endpoint on Compute Engine, typically listening on port 80, to which external applications can send POST requests containing input data.

4. **Invocation**: To invoke our deployed service, external applications make HTTP POST requests to the specified endpoint, passing the required input data as a string payload. The service then processes the request, generates predictions, and returns the delay prediction as JSON responses:

To post a single data point for prediction from the command line:
`curl -X 'POST' \
  'http://<ENDPOINT_IP>/predict?input_data=%22%5B8000207.0%2C0%2C6%2C6%2C16%2C15%2C9 [ more data ] C1%2C650%2C1%5D%22' \
  -H 'accept: application/json' \
  -d ''`
Response looks like the following:
`{"input":"[ input data ]","message":"OK","status-code":200,"prediction":[{"delay":0.130349263548851}]}`

By deploying our model in the cloud with Docker and FastAPI, we created an accessible inference service with easy ways for increasing the performance short term and the option to make it fully scalable in the future.

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

Monitoring was implemented by collecting inference data (saving time, input and output) within the inference application and using evidently to create respective reports. Monitoring is available through a fastapi application that is deployed to the google cloud. We provide two endpoints, each delivering a HTML output for visual inspection, that deal with two main components of our application:

1. **Data**: The monitoring report for data is available at the "/monitoring" endpoint and involves checking general data quality and detection of data drifts. It uses a sample of 1000 training data points and the collected inference data saved to a storage bucket in the google cloud.

2. **Model**: At the endpoint "/monitoring-tests" a report assessing the model quality via different tests is available. Here we similarly use 1000 data points to check the models performance w.r.t. to different common measures like e.g. Precision, Recall or F1.

Due to time constraints we could not provide a more extensive monitoring for our application, though there are more details that would help to ensure the model and application quality:

- Monitoring general application traffic: Though application traffic could be inferred from the collected inference data, there are more convinient frameworks available, like opentelemetry for extracting and Signoz for visualizing relevant telemetry data. This would help us to gain better insights in usage statistics of our application, indicating e.g. malicious use or a need for scaling.
- Monitoring live performance: An interesting addition to monitoring the model's performance on the available data would be to track the performance on the input data from the inference application. A possible way to achive this would be to collect further da from inference, i.e. like/dislike of the prediction. With this target drifts or general underperfomance of the model could be detected.


### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

As of now, we've utilized approximately 30€ of our available credits, primarily attributed to Compute Engine, where we are using rather performant machines. The cost can mosly be attributedto instances inadvertently running overnight due to automated deployments. This is a major downside of Compute Engine, since it also costs money when you are not actively using the resources. This is good for training, where we need high performance for a short time, but more inconvenient for inference, at least as long as inference is done on a small scale.
However, our financial resources were robust, with over 300€ at our disposal. To ensure not running out of budget, we've implemented budgeting alerts and set cutoffs at 50€ as a precautionary measure.

It's worth noting that the actual cost of operations during periods when we actively required resources amounted to less than 5€ by our estimations.

[Our Cost Overview](figures/costs.png)


## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

[System Architecture](figures/architecture2.png)

The architecture primarily revolves around cloud-based infrastructure, focusing on machine learning model development and deployment. The core components and steps in the architecture are:

1. **GitHub Repository**: The project's codebase is hosted on GitHub, providing version control and collaboration tools, as well as continuous integration.

2. **Continuous Integration (CI)**: GitHub Actions automates code testing and the container building as well as deployment processes, ensuring code quality and efficient development, where we do not have to worry about deploying each small improvement to our code individually.

3. **Docker Images**: Docker containers house the machine learning model, as already described before, enabling consistency and portability. They are stored and versioned on DockerHub.


4. **Google Cloud**

- Compute Engine: VM instances in Google Compute Engine are utilized for both model training and inference and for monitoring data and model. These instances are configured to run Docker containers, encapsulating the model or application and dependencies. FastAPI is employed to serve the model and monitoring as an API, allowing external systems to make HTTP requests for predictions.
- Data and Model Storage: Google Cloud Storage is used for storing datasets, trained models, inference data, and various project artifacts. It provides secure and scalable object storage.

5. **Document Flow**:
- Data flows:
	- from Google Cloud Storage to the Docker containers during training
	- from Google Cloud Storage to the FastAPI service for monitoring when generating a requested report
	- from external applications to the FastAPI service for inference and to the Google Cloud Storage where it is stored for monitoring purposes
- Models that result from training are stored in Google Cloud Storage and used in the inference container as well as for some monitoring.

6. **Monitoring**: Monitoring is available through a FastAPI application deployed in the cloud and can be similary to the inference assessed via HTTP requests

This architecture is designed to facilitate efficient and (in the future) scalable model development, training, and deployment. It leverages cloud services and automation to streamline the machine learning operations. Furthermore, it enables maintaining cost control, both through using easy to plan, but flexible enough services and by enabling easy switching of the compute platform, should one service get too expensive.

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

The project presented several challenges throughout its lifecycle, which required strategic planning. Since the development time was limited, we had to focus on the most relevant aspects of our planned architecture. Nevertheless, we did not encounter many unexpected challenges. One major challenge that cost us a lot of time, the authentication for deploying services to and in the cloud, was expected beforehand. Most other issues were caused by dependencies, different development environments on different operating systems on laptops and in the cloud and managing files and configurations.


1. **Complex Cloud Setup**: Configuring and managing cloud infrastructure, including Google Compute Engine, posed challenges. Especially challenging (and expected) was granting the virtual machines in the cloud access to the cloud storage buckets. The virtual machine, next to having an attached service account, also needs to get a scope, which was not mentioned everywhere. Overcoming this, we had to rely on documented procedures by google, since some of the dtu material was lacking some information necessary for our concrete setup, and searched for similar problems on community forums.

2. **Debugging in Cloud**: Debugging issues in a cloud-based environment was challenging. We mitigated this by SSHing into virtual machines for direct troubleshooting and carefully examining logs and error messages. Especially problematic was debugging the GitHub workflows, since this option is not available in this case.

3. **Hyperparameter Tuning**: Fine-tuning hyperparameters costs development time. Since our model was not converging at first and since we want our model to be able to automatically retrain itself with new data, we are using SMAC for multi-objective hyperparameter tuning. Writing the code for this was time consuming, since debugging was difficult due to this multi-threaded and configuration-heavy application.
Problematic was also to install the dependency swig for the tuning library SMAC on our local development environment and in the Docker container.

5. **Code Integration and CI/CD**: Integrating various components of the project and ensuring a seamless CI/CD pipeline demanded time and effort. We iteratively refined our GitHub Actions workflows, optimized code quality checks, and conducted thorough testing to streamline the process.

In overcoming these challenges, communication and collaboration among the two team members played a crucial role. Regular meetings, talking about current issues and an agile approach to problem-solving were key. Our commitment to continuous improvement allowed us to address project hurdles effectively.

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

***
Maya Köhnen:

- FastAPI Inference Application
- FastAPI Monitoring Application and required data processing
- Configurability and Reproducibility
- Unit tests
- Code Quality (typing, linting, documentation, refactoring)

***

Jan Anders:

- Dataset creation and management
- Training and hyperparameter tuning
- Dockerization
- GitHub Actions Workflows
- Cloud management and authorizations
***
