"""
Farnsworth Remotion Integration - Programmatic Video Generation.

"Lights, Camera, React!"

This module enables Farnsworth to generate video scripts (Remotion/React)
and render them using the Remotion CLI.
"""

import subprocess
import os
import json
from pathlib import Path
from loguru import logger
from typing import Dict, Any

class RemotionVideoSkill:
    def __init__(self, workspace_path: str = "./remotion_workspace"):
        self.workspace = Path(workspace_path)
        self.workspace.mkdir(parents=True, exist_ok=True)
        
    async def create_video_project(self, name: str):
        """Initialize a new remotion project template."""
        logger.info(f"Remotion: Creating project {name}")
        # In real life: npx create-remotion@latest --template ...
        # For now, we assume a base template exists or we write a basic index.tsx
        pass

    async def render_video(self, composition_id: str, props: Dict[str, Any], output_path: str):
        """
        Render a remotion composition to an MP4 file.
        Requires remotion installed in the workspace.
        """
        logger.info(f"Remotion: Rendering {composition_id}")
        
        # Serialize props to JSON
        props_file = self.workspace / "props.json"
        with open(props_file, "w") as f:
            json.dump(props, f)
            
        cmd = [
            "npx", "remotion", "render", 
            composition_id, 
            output_path, 
            "--props", str(props_file)
        ]
        
        try:
            # We run this in the background / sub-task
            process = subprocess.Popen(cmd, cwd=str(self.workspace), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                logger.success(f"Remotion: Render complete -> {output_path}")
                return True
            else:
                logger.error(f"Remotion: Render failed: {stderr.decode()}")
                return False
        except Exception as e:
            logger.error(f"Remotion: Execution error: {e}")
            return False

    def generate_component_code(self, narrative: str) -> str:
        """
        Generate React/Remotion code based on a narrative summary.
        (Usually called by an LLM)
        """
        return f"""
import {{Composition}} from 'remotion';
import {{MyComp}} from './MyComp';

export const RemotionVideo = () => {{
  return (
    <>
      <Composition
        id="Main"
        component={{MyComp}}
        durationInFrames={{300}}
        fps={{30}}
        width={{1920}}
        height={{1080}}
        defaultProps={{{{
          text: "{narrative}"
        }}}}
      />
    </>
  );
}};
"""

# Global instance
remotion_skill = RemotionVideoSkill()
