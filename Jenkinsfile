pipeline {
    agent any

    environment {
        IMAGE_NAME = 'aditya1357/dsa_visualize'
    }

    stages {
        stage('Clone') {
            steps {
                git branch: 'main', url: 'https://github.com/dredninja/DSA_Visualizer'
            }
        }

        stage('Run Tests') {
            steps {
                bat 'python manage.py test'
            }
        }

        stage('Build Docker Image') {
            steps {
                bat "docker build -t %IMAGE_NAME% ."
            }
        }

        stage('Push to Docker Hub') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'aditya1357', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                    bat """
                    echo %DOCKER_PASS% | docker login -u %DOCKER_USER% --password-stdin
                    docker push %IMAGE_NAME%
                    """
                }
            }
        }

        stage('Deploy Container') {
            steps {
                bat 'docker run -d -p 8000:8000 %IMAGE_NAME%'
            }
        }
    }
}
