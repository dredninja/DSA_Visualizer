pipeline {
    agent any

    environment {
        IMAGE_NAME = 'yourdockerhubusername/django-devops-demo'
    }

    stages {
        stage('Clone') {
            steps {
                git 'https://github.com/yourusername/django-devops-demo.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Run Tests') {
            steps {
                sh 'python manage.py test'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t $IMAGE_NAME .'
            }
        }

        stage('Push to Docker Hub') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub-creds', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                    sh """
                        echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin
                        docker push $IMAGE_NAME
                    """
                }
            }
        }

        stage('Deploy Container') {
            steps {
                sh 'docker run -d -p 8000:8000 $IMAGE_NAME'
            }
        }
    }
}
