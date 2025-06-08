#!/bin/bash

# Install required packages
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="your-openai-api-key-here"

# Run the agent
python screen_activity_agent.py