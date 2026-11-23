variable "aws_region" {
    description = "AWS region to deploy into"
    type = string
    default = "us-east-1"
}

variable "instance_type" {
    description = "EC2 size"
    type = string
    default = "t3.large"
}

variable "key_name" {
    description = "Existing AWS EC2 pair name SSH access key"
    type = string
}

variable "repo_url" {
    description = "Git URL of MULTI-Agent"
    type = string
}

variable "allowed_cidr" {
    description = "IP CIDR allowed to access Streamlit , API ,and SSH"
    type = string
    default = "0.0.0.0/0"
}