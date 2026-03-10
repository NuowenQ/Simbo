<div align="center">

# Simbo

**Autonomous AI Assistant for ROS/Gazebo Simulation Development**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![LangGraph](https://img.shields.io/badge/Powered%20by-LangGraph-orange?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![ROS](https://img.shields.io/badge/ROS-1%20%7C%202-22314E?style=flat-square&logo=ros)](https://www.ros.org/)

Simbo is an autonomous multi-agent AI platform that writes, edits, builds, and validates ROS/Gazebo code — so you can focus on the robot, not the boilerplate.

</div>

---

## What is Simbo?

Simbo is not a chatbot that gives you code snippets to copy-paste. It is an autonomous agent that **acts directly on your workspace** — the same way tools like Cursor or Claude Code act on your codebase.

Give Simbo a task in plain English. It will:

1. Analyze your ROS workspace, packages, topics, and joints
2. Plan and implement the required code changes
3. Build the workspace and validate everything compiles
4. Iterate and self-correct until the task is complete

Simbo ships with four specialized modules, each powered by an independent LangGraph agent optimized for its domain.

---

## Modules

### `Program` — Autonomous ROS Developer

The core Simbo agent. Describe what you need; Simbo writes the code, builds the workspace, runs ROS commands, and verifies the result — without leaving the conversation.

**Stage-Based Controller Generation**

Simbo enforces a disciplined, three-stage controller development process that mirrors best practices in robotics engineering:

| Stage | What It Does | Why It Matters |
|-------|-------------|----------------|
| **Stage 1 — Logging Only** | Creates a minimal ROS node that subscribes to sensor topics and logs received data | Verifies the node launches and topic connections are correct before any motion commands |
| **Stage 2 — Open-Loop Commands** | Adds constant velocity or position commands with no feedback | Confirms command publishing works correctly before introducing control loops |
| **Stage 3 — Closed-Loop Control** | Implements PID or feedback controllers using real sensor data | Ensures safe, validated progression to full closed-loop behavior |

Simbo never skips stages. Each stage is built, compiled, and verified before the next begins.

**Supported controller types:** velocity, position, joint trajectory, teleop

---

### `World` — Gazebo World Design Agent

Describe the simulation environment you need in plain English. The World agent searches a curated database of open-source Gazebo worlds using semantic RAG retrieval, selects the best match, downloads all required models, and places everything into your workspace — automatically.

- Extracts environment constraints (indoor/outdoor, scale, terrain, semantic features) from your description
- Symbolic filtering + vector search to rank candidate worlds
- Downloads and places world files into a properly structured ROS package (`simbo_worlds`)
- Automatically updates your existing simulation launch files — no manual steps

**Sources:** AWS RoboMaker, OSRF Gazebo Models, TurtleBot3 Simulations, Clearpath, and more.

---

### `Map Generator` — 2D Image to Gazebo World

Upload any top-down floor plan or map image. Simbo extracts edges and geometric shapes using computer vision and generates a ready-to-use `.world` file for Gazebo.

- Powered by OpenCV edge detection and contour analysis
- Outputs a valid Gazebo SDF world file
- Works with floor plans, satellite maps, or hand-drawn sketches

---

### `Robot` — Robot Model Manager

Configure and manage robot models in your workspace. Inspect URDF/Xacro files, review joint definitions, and prepare robot configurations for use with the Program and World agents.

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                      Streamlit UI                          │
│          Program │ World │ Map Generator │ Robot           │
├────────────────────────────────────────────────────────────┤
│                   LangGraph Agents                         │
│                                                            │
│   SimulationAgent              WorldDesignAgent            │
│   ┌──────────┐                 ┌──────────────────────┐    │
│   │  agent   │◄──────loop──────│  extract constraints │    │
│   └────┬─────┘                 │  search database     │    │
│        │ tool calls            │  download & place    │    │
│   ┌────▼─────┐                 │  update launch files │    │
│   │  tools   │                 └──────────────────────┘    │
│   └────┬─────┘                                             │
│        │                                                   │
│   ┌────▼──────────┐                                        │
│   │ validate_comp │  (build ✓ node loads ✓ topics ✓)       │
│   └───────────────┘                                        │
├────────────────────────────────────────────────────────────┤
│                       Tool Layer                           │
│  Workspace Tools │ File Tools │ Shell Tools │ World Tools  │
│  detect_ros_version    read_file       run_command         │
│  analyze_workspace     write_file      build_ros_workspace │
│  list_packages         edit_file       check_ros_topics    │
│  find_launch_files     search_in_files check_ros_nodes     │
└────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.9+
- OpenAI API key
- ROS 1 or ROS 2 workspace (optional — required for controller generation and workspace analysis)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/simbo.git
cd simbo

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure your OpenAI API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=<your-key>
```

For the **Map Generator** module, install the optional computer vision dependencies:

```bash
pip install opencv-python numpy
```

### Development Installation

```bash
pip install -e ".[dev]"
```

---

## Usage

### Launch the UI

```bash
streamlit run src/simbo/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Programmatic Usage

**Simulation Agent (code generation):**

```python
from simbo.agents import create_simulation_graph

agent = create_simulation_graph(
    model_name="gpt-4-turbo-preview",
    api_key="your-openai-api-key"
)

result = agent.invoke(
    user_input="Generate a velocity controller for my differential drive robot",
    workspace_path="/path/to/your/ros_workspace"
)

for message in result["messages"]:
    print(message.content)
```

**World Design Agent:**

```python
from simbo.agents.world_agent import create_world_agent

agent = create_world_agent(api_key="your-openai-api-key")

result = agent.invoke(
    user_input="I need a small indoor office environment for navigation testing",
    workspace_path="/path/to/your/ros_workspace"
)
```

### Example Prompts

**Controller generation:**
- `"Generate a velocity controller for my robot named 'turtlebot'"`
- `"Create a position controller that navigates to goal poses"`
- `"Generate a joint trajectory controller for a 6-DOF arm"`
- `"Write a teleop keyboard controller for my mobile base"`

**World design:**
- `"I need a large outdoor environment with uneven terrain for off-road testing"`
- `"Find me an indoor warehouse world for AMR navigation"`
- `"Set up a hospital environment for service robot simulation"`

**Workspace analysis:**
- `"Analyze my ROS workspace and list all packages"`
- `"Find all files that subscribe to /cmd_vel"`

---

## Project Structure

```
Simbo/
├── src/
│   └── simbo/
│       ├── app.py                    # Streamlit entry point
│       ├── agents/
│       │   ├── simulation_agent.py   # Autonomous ROS coding agent
│       │   └── world_agent.py        # World Design Agent
│       ├── tools/
│       │   ├── workspace_tools.py    # ROS workspace analysis
│       │   ├── file_tools.py         # File read/write/edit operations
│       │   ├── shell_tools.py        # Shell and ROS command execution
│       │   ├── code_tools.py         # Code generation utilities
│       │   └── world_tools.py        # World search, download, placement
│       ├── views/
│       │   ├── home.py               # Main navigation page
│       │   ├── program.py            # Program module UI
│       │   ├── world.py              # World module UI
│       │   ├── robot.py              # Robot module UI
│       │   ├── map_generator.py      # Map Generator UI
│       │   └── components.py         # Shared UI components
│       └── utils/
│           ├── state.py              # Shared state definitions
│           └── map_generator.py      # CV-based map processing
├── config/
│   └── settings.yaml                 # Configuration
├── requirements.txt
├── setup.py
├── pyproject.toml
└── README.md
```

---

## Configuration

Edit `config/settings.yaml` to customize model parameters, workspace analysis limits, default controller values, and UI preferences.

```yaml
openai:
  model: "gpt-4-turbo-preview"
  temperature: 0.1
  max_tokens: 4096

agent:
  max_iterations: 25
  enable_streaming: true

code_generation:
  default_params:
    linear_speed: 0.5
    angular_speed: 1.0
    publish_rate: 10.0
```

---

<<<<<<< HEAD
### Stage 2: Multi-Agent Architecture
- Manager Agent for orchestration
- World Design Agent (completed)
- Robot Model Agent
- Programming Agent (completed)
=======
## Roadmap
>>>>>>> 17a5da4 (Finalize readme)

### Current Release
- [x] Autonomous ROS code generation (controllers, launch files, packages)
- [x] ROS 1 and ROS 2 support (ament_python)
- [x] Stage-based controller development (log → open-loop → closed-loop)
- [x] Deterministic validation (build + topics + nodes + executables)
- [x] World Design Agent with semantic RAG retrieval
- [x] 2D map-to-Gazebo-world conversion
- [x] Token-optimized conversation summarization
- [x] Streamlit multi-module UI

### Upcoming
- [ ] Robot Model Agent (URDF/SDF generation)
- [ ] Full simulation pipeline (world + robot + controller in one command)
- [ ] Support for additional AI providers
- [ ] Ignition/Gazebo Harmonic world support

---

## Contributing

<<<<<<< HEAD
Contributions are welcome!
=======
Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---
>>>>>>> 17a5da4 (Finalize readme)

## Acknowledgments

- Powered by OpenAI GPT-4 API
- UI built with [Streamlit](https://streamlit.io/)
- World database sourced from AWS RoboMaker, OSRF, ROBOTIS, and Clearpath open-source repositories
