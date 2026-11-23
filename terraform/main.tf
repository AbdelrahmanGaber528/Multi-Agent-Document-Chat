# Fetch latest Ubuntu 24.04 LTS AMI
data "aws_ami" "ubuntu" {
    most_recent = true
    owners      = ["099720109477"] # Canonical

    filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
    }
}

# --- Networking ---
resource "aws_vpc" "main" {
    cidr_block           = "10.0.0.0/16"
    enable_dns_support   = true
    enable_dns_hostnames = true

    tags = {
    Name = "agent-multi-chat-vpc"
    }
}

resource "aws_internet_gateway" "igw" {
    vpc_id = aws_vpc.main.id

    tags = {
    Name = "agent-multi-chat-igw"
    }
}

resource "aws_subnet" "public" {
    vpc_id                  = aws_vpc.main.id
    cidr_block              = "10.0.1.0/24"
    map_public_ip_on_launch = true

    tags = {
    Name = "agent-multi-chat-public-subnet"
    }
}

resource "aws_route_table" "public" {
    vpc_id = aws_vpc.main.id

    route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
    }

    tags = {
    Name = "agent-multi-chat-public-rt"
    }
}

resource "aws_route_table_association" "public" {
    subnet_id      = aws_subnet.public.id
    route_table_id = aws_route_table.public.id
}

# --- Security Group ---
resource "aws_security_group" "app_sg" {
    name        = "rag-chat-sg"
    description = "Allow SSH, FastAPI (8000), and Streamlit (8501)"
    vpc_id      = aws_vpc.main.id

    ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
    }

    ingress {
    description = "FastAPI Backend"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
    }

    ingress {
    description = "Streamlit UI"
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
    }

    egress {
    description = "Allow all outbound (LLM APIs, HuggingFace, Docker Hub)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    }

    tags = {
    Name = "agent-multi-chat-sg"
    }
}

# --- Compute & Persistent Storage ---
resource "aws_instance" "app_server" {
    ami                    = data.aws_ami.ubuntu.id
    instance_type          = var.instance_type
    subnet_id              = aws_subnet.public.id
    vpc_security_group_ids = [aws_security_group.app_sg.id]
    key_name               = var.key_name

    root_block_device {
    volume_size           = 30
    volume_type           = "gp3"
    delete_on_termination = false
    }

    user_data = templatefile("${path.module}/user_data.sh", {
    repo_url = var.repo_url
    })

    tags = {
    Name = "agent-multi-chat-server"
    }
}