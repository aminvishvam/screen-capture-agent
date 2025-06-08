import time
import cv2
import numpy as np
import pyautogui
import psutil
import json
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass
from langchain.agents import BaseSingleActionAgent
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
import threading
import queue

@dataclass
class ActivityEvent:
    timestamp: str
    event_type: str
    details: Dict[str, Any]
    screenshot_path: str = None

class ScreenCaptureInput(BaseModel):
    interval: int = Field(default=5, description="Screenshot interval in seconds")

class ScreenCaptureTool(BaseTool):
    name = "screen_capture"
    description = "Capture screenshots at specified intervals"
    args_schema = ScreenCaptureInput
    
    def __init__(self):
        super().__init__()
        self.last_screenshot = None
        self.screenshot_count = 0
    
    def _run(self, interval: int = 5) -> str:
        try:
            screenshot = pyautogui.screenshot()
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            screenshot.save(filename)
            
            # Convert to numpy array for comparison
            current_screenshot = np.array(screenshot)
            
            # Check if screen content has changed significantly
            if self.last_screenshot is not None:
                diff = cv2.absdiff(current_screenshot, self.last_screenshot)
                change_percentage = (np.sum(diff) / diff.size) * 100
                
                if change_percentage > 5:  # Threshold for significant change
                    self.last_screenshot = current_screenshot
                    self.screenshot_count += 1
                    return f"Screenshot captured: {filename} (Change detected: {change_percentage:.2f}%)"
                else:
                    return "No significant screen change detected"
            else:
                self.last_screenshot = current_screenshot
                self.screenshot_count += 1
                return f"Initial screenshot captured: {filename}"
                
        except Exception as e:
            return f"Error capturing screenshot: {str(e)}"

class MouseActivityInput(BaseModel):
    duration: int = Field(default=60, description="Duration to monitor mouse activity in seconds")

class MouseActivityTool(BaseTool):
    name = "mouse_activity"
    description = "Monitor mouse movements and clicks"
    args_schema = MouseActivityInput
    
    def _run(self, duration: int = 60) -> str:
        try:
            start_time = time.time()
            mouse_events = []
            last_position = pyautogui.position()
            
            while time.time() - start_time < duration:
                current_position = pyautogui.position()
                
                if current_position != last_position:
                    mouse_events.append({
                        'timestamp': datetime.utcnow().isoformat(),
                        'type': 'mouse_move',
                        'position': current_position,
                        'distance': ((current_position.x - last_position.x)**2 + 
                                   (current_position.y - last_position.y)**2)**0.5
                    })
                    last_position = current_position
                
                time.sleep(0.1)  # Check every 100ms
            
            return f"Monitored {len(mouse_events)} mouse movements in {duration} seconds"
            
        except Exception as e:
            return f"Error monitoring mouse activity: {str(e)}"

class ProcessMonitorInput(BaseModel):
    top_n: int = Field(default=5, description="Number of top processes to monitor")

class ProcessMonitorTool(BaseTool):
    name = "process_monitor"
    description = "Monitor running processes and their resource usage"
    args_schema = ProcessMonitorInput
    
    def _run(self, top_n: int = 5) -> str:
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    proc_info = proc.info
                    if proc_info['cpu_percent'] > 0:
                        processes.append(proc_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Sort by CPU usage
            processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
            top_processes = processes[:top_n]
            
            result = "Top active processes:\n"
            for proc in top_processes:
                result += f"- {proc['name']} (PID: {proc['pid']}): CPU {proc['cpu_percent']:.1f}%, Memory {proc['memory_percent']:.1f}%\n"
            
            return result
            
        except Exception as e:
            return f"Error monitoring processes: {str(e)}"

class ActivityAnalysisInput(BaseModel):
    activity_data: str = Field(description="JSON string of activity data to analyze")

class ActivityAnalysisTool(BaseTool):
    name = "activity_analysis"
    description = "Analyze user activity patterns and provide insights"
    args_schema = ActivityAnalysisInput
    
    def _run(self, activity_data: str) -> str:
        try:
            data = json.loads(activity_data)
            
            # Basic analysis
            total_events = len(data)
            event_types = {}
            
            for event in data:
                event_type = event.get('type', 'unknown')
                event_types[event_type] = event_types.get(event_type, 0) + 1
            
            analysis = f"Activity Analysis:\n"
            analysis += f"Total events: {total_events}\n"
            analysis += f"Event breakdown:\n"
            
            for event_type, count in event_types.items():
                percentage = (count / total_events) * 100
                analysis += f"- {event_type}: {count} ({percentage:.1f}%)\n"
            
            return analysis
            
        except Exception as e:
            return f"Error analyzing activity: {str(e)}"

class ScreenActivityAgent(BaseSingleActionAgent):
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.activity_log = []
        self.monitoring = False
        self.monitor_thread = None
        
    def plan(self, intermediate_steps, **kwargs):
        user_input = kwargs.get('input', '')
        
        if 'stop' in user_input.lower():
            self.stop_monitoring()
            return AgentFinish(
                return_values={'output': 'Monitoring stopped'},
                log='Stopped monitoring user activity'
            )
        
        if 'start' in user_input.lower() or 'monitor' in user_input.lower():
            self.start_monitoring()
            return AgentFinish(
                return_values={'output': 'Started monitoring user activity'},
                log='Started monitoring user activity'
            )
        
        if 'screenshot' in user_input.lower():
            return AgentAction(
                tool='screen_capture',
                tool_input={'interval': 5},
                log='Taking screenshot'
            )
        
        if 'mouse' in user_input.lower():
            return AgentAction(
                tool='mouse_activity',
                tool_input={'duration': 30},
                log='Monitoring mouse activity'
            )
        
        if 'process' in user_input.lower():
            return AgentAction(
                tool='process_monitor',
                tool_input={'top_n': 5},
                log='Monitoring processes'
            )
        
        if 'analyze' in user_input.lower():
            activity_json = json.dumps(self.activity_log)
            return AgentAction(
                tool='activity_analysis',
                tool_input={'activity_data': activity_json},
                log='Analyzing activity data'
            )
        
        return AgentFinish(
            return_values={'output': 'Available commands: start, stop, screenshot, mouse, process, analyze'},
            log='Provided help information'
        )
    
    def start_monitoring(self):
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._continuous_monitor)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
    
    def _continuous_monitor(self):
        while self.monitoring:
            try:
                # Capture screenshot
                screenshot_result = self.tools['screen_capture']._run(interval=10)
                
                # Monitor processes
                process_result = self.tools['process_monitor']._run(top_n=3)
                
                # Log activity
                activity_event = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'type': 'monitoring_cycle',
                    'screenshot_result': screenshot_result,
                    'process_info': process_result
                }
                
                self.activity_log.append(activity_event)
                
                # Keep only last 100 events
                if len(self.activity_log) > 100:
                    self.activity_log = self.activity_log[-100:]
                
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(5)
    
    @property
    def input_keys(self):
        return ['input']

def create_screen_activity_agent():
    """Create and configure the screen activity tracking agent"""
    
    # Initialize LLM (you'll need to set your OpenAI API key)
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
        # openai_api_key="your-api-key-here"
    )
    
    # Initialize tools
    tools = [
        ScreenCaptureTool(),
        MouseActivityTool(),
        ProcessMonitorTool(),
        ActivityAnalysisTool()
    ]
    
    # Create agent
    agent = ScreenActivityAgent(llm, tools)
    
    return agent

def main():
    """Main function to run the screen activity agent"""
    print("Screen Activity Tracking Agent")
    print("=" * 40)
    print("Available commands:")
    print("- 'start' or 'monitor': Start continuous monitoring")
    print("- 'stop': Stop monitoring")
    print("- 'screenshot': Take a screenshot")
    print("- 'mouse': Monitor mouse activity")
    print("- 'process': Check running processes")
    print("- 'analyze': Analyze collected activity data")
    print("- 'quit': Exit the program")
    print()
    
    agent = create_screen_activity_agent()
    
    while True:
        try:
            user_input = input(f"[{datetime.utcnow().strftime('%H:%M:%S')}] Enter command: ").strip()
            
            if user_input.lower() == 'quit':
                agent.stop_monitoring()
                break
            
            if user_input:
                # Execute agent action
                result = agent.plan([], input=user_input)
                
                if isinstance(result, AgentAction):
                    tool = agent.tools[result.tool]
                    tool_result = tool._run(**result.tool_input)
                    print(f"Result: {tool_result}")
                elif isinstance(result, AgentFinish):
                    print(f"Agent: {result.return_values['output']}")
        
        except KeyboardInterrupt:
            print("\nStopping agent...")
            agent.stop_monitoring()
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()