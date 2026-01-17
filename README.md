# Simbo

**Multi-Agent Gazebo Simulation Assistant** powered by LangChain and LangGraph.

Simbo is an AI-powered assistant that helps roboticists generate control programs for ROS/Gazebo simulations. It analyzes your ROS workspace, understands your existing code, and helps you develop controllers, launch files, and other simulation components.

## Features

- **Workspace Analysis**: Automatically detects ROS version (ROS1/ROS2), Gazebo version, and analyzes your packages
- **Code Generation**: Generates velocity controllers, position controllers, joint trajectory controllers, and teleop controllers
- **Intelligent Assistance**: Uses GPT-4 to understand your requirements and generate appropriate code
- **Interactive UI**: Streamlit-based interface for easy interaction
- **LangGraph Workflow**: Sophisticated agent workflow for multi-step tasks

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit UI                           │
├─────────────────────────────────────────────────────────────┤
│                   LangGraph Workflow                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Process  │→ │  Check   │→ │ Analyze  │→ │ Generate │   │
│  │  Input   │  │Workspace │  │   Env    │  │ Response │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│                              ↓                              │
│                       ┌──────────┐                          │
│                       │  Tools   │                          │
│                       └──────────┘                          │
├─────────────────────────────────────────────────────────────┤
│                     Tool Layer                              │
│  ┌────────────────┐  ┌────────────────┐                    │
│  │ Workspace Tools│  │   Code Tools   │                    │
│  │ - detect_ros   │  │ - read_file    │                    │
│  │ - analyze_ws   │  │ - write_file   │                    │
│  │ - list_pkgs    │  │ - search_code  │                    │
│  │ - find_launch  │  │ - gen_ctrl     │                    │
│  └────────────────┘  └────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.9+
- OpenAI API key
- (Optional) ROS/ROS2 workspace for full functionality

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/simbo.git
cd simbo
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Usage

### Running the Streamlit App

```bash
streamlit run src/simbo/app.py
```

Then open your browser to `http://localhost:8501`.

### Using the Agent Programmatically

```python
from simbo.agents import create_simulation_graph

# Create the agent
agent = create_simulation_graph(
    model_name="gpt-4-turbo-preview",
    api_key="your-openai-api-key"
)

# Invoke with a workspace
result = agent.invoke(
    user_input="Generate a velocity controller for my differential drive robot",
    workspace_path="/path/to/your/ros_workspace"
)

# Access the response
for message in result["messages"]:
    print(message.content)
```

### Example Commands

- **Analyze workspace**: "Analyze my ROS workspace and show me the packages"
- **Generate controller**: "Generate a velocity controller for my robot named 'turtlebot'"
- **Position control**: "Create a position controller that navigates to goal poses"
- **Joint control**: "Generate a joint trajectory controller for a 6-DOF arm"
- **Search code**: "Find all files that subscribe to /cmd_vel"

## Project Structure

```
Simbo/
├── src/
│   └── simbo/
│       ├── __init__.py
│       ├── app.py              # Streamlit UI
│       ├── agents/
│       │   ├── __init__.py
│       │   └── simulation_agent.py  # Main LangGraph agent
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── workspace_tools.py   # ROS workspace tools
│       │   └── code_tools.py        # Code generation tools
│       └── utils/
│           ├── __init__.py
│           └── state.py             # State definitions
├── config/
│   └── settings.yaml           # Configuration
├── tests/
├── requirements.txt
├── setup.py
├── pyproject.toml
└── README.md
```

## Configuration

Edit `config/settings.yaml` to customize:

- OpenAI model and parameters
- Workspace analysis settings
- Code generation defaults
- UI preferences

## Stage 1 Features (Current)

- [x] ROS workspace detection and analysis
- [x] ROS1 and ROS2 support
- [x] Velocity controller generation
- [x] Position controller generation
- [x] Joint trajectory controller generation
- [x] Teleop controller generation
- [x] Code search and analysis
- [x] Interactive Streamlit UI
- [x] LangGraph workflow orchestration

## Future Stages

### Stage 2: Multi-Agent Architecture
- Manager Agent for orchestration
- World Design Agent
- Robot Model Agent
- Programming Agent

### Stage 3: Full Simulation Pipeline
- URDF/SDF generation
- World file generation
- Launch file generation
- Full simulation setup

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built with [LangChain](https://langchain.com/) and [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by OpenAI GPT-4
- UI built with [Streamlit](https://streamlit.io/)
