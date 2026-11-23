output "public_ip" {
    description = "Public IP of EC2 instance"
    value = aws_instance.app_server.public_ip
}

output "streamlit_url" {
    description = "Streamlit UI URL"
    value = "http://${aws_instance.app_server.public_ip}:8501"
}


output "fastapi_docs_url"{
    description = "FastAPI swagger docs URL"
    value = "http://${aws_instance.app_server.public_ip}:8000/docs"
}