pipeline {
    agent any

    environment {
        IMAGE_NAME = 'aditya1357/DSA_Visualizer'
    }

    stages {
        stage('Clone') {
            steps {
                git branch:'main', url: 'https://github.com/dredninja/DSA_Visualizer'
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
                withCredentials([usernamePassword(credentialsId: 'aditya1357', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
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
