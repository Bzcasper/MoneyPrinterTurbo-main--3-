#!/usr/bin/env python3
"""
Claude CLI - Rule Enforcement and RL Integration for MoneyPrinterTurbo Analysis
Following Claude_General.prompt.md instructions
"""

import argparse
import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

class ClaudeCLI:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.todo_file = self.project_root / "TODO.md"
        self.env_file = self.project_root / "credentials.env"
        
    def rule_check(self, task=None):
        """Verify TODO.md updates, branch creation, env file presence"""
        issues = []
        
        # Check TODO.md exists
        if not self.todo_file.exists():
            issues.append("TODO.md missing")
        
        # Check credentials.env exists and is locked
        if not self.env_file.exists():
            issues.append("credentials.env missing")
        elif oct(os.stat(self.env_file).st_mode)[-3:] != '400':
            issues.append("credentials.env not locked (should be chmod 400)")
            
        # Check git branch
        try:
            result = subprocess.run(['git', 'branch', '--show-current'], 
                                  capture_output=True, text=True)
            current_branch = result.stdout.strip()
            if current_branch == 'main' or current_branch == 'master':
                issues.append("Still on main/master branch - should create feature branch")
        except:
            issues.append("Git not available or not in git repo")
            
        if issues:
            print("❌ RULE CHECK FAILED:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("✅ RULE CHECK PASSED")
            return True
    
    def branch_init(self, branch_name, description=""):
        """Create new git branch and log to TODO.md"""
        try:
            # Create and checkout new branch
            subprocess.run(['git', 'checkout', '-b', branch_name], check=True)
            print(f"✅ Created branch: {branch_name}")
            
            # Log to TODO.md
            self.todo_update(f"Branch created: {branch_name}", "created", description)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create branch: {e}")
            return False
    
    def todo_update(self, task, status, verify=""):
        """Update TODO.md with task status"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if not self.todo_file.exists():
            content = "# MoneyPrinterTurbo Analysis TODO\n\n"
        else:
            content = self.todo_file.read_text()
        
        entry = f"- [{timestamp}] {task}: {status.upper()}"
        if verify:
            entry += f" (verified: {verify})"
        entry += "\n"
        
        content += entry
        self.todo_file.write_text(content)
        print(f"✅ TODO updated: {task} -> {status}")
    
    def env_lock(self, key=None, value=None, lock=True):
        """Create/update credentials.env and lock it"""
        if key and value:
            # Add or update key-value pair
            content = ""
            if self.env_file.exists():
                content = self.env_file.read_text()
            
            lines = content.split('\n')
            found = False
            for i, line in enumerate(lines):
                if line.startswith(f"{key}="):
                    lines[i] = f"{key}={value}"
                    found = True
                    break
            
            if not found:
                lines.append(f"{key}={value}")
            
            content = '\n'.join(lines)
            self.env_file.write_text(content)
        
        if lock:
            os.chmod(self.env_file, 0o400)  # Read-only for owner
            print(f"✅ Environment file locked: {self.env_file}")
    
    def sub_agent_spawn(self, task_description, output_file="prompt.xml"):
        """Generate optimized sub-agent prompt"""
        prompt = f"""<sub_agent_task>
Task: {task_description}
Requirements:
- Minimize token usage
- Maximize clarity and effectiveness
- Follow structured output format
- Include error handling
- Provide measurable success criteria

Output format: Structured XML or JSON with clear sections
Success criteria: Task completion verification
Error handling: Graceful failure with specific error messages
</sub_agent_task>"""
        
        Path(output_file).write_text(prompt)
        print(f"✅ Sub-agent prompt generated: {output_file}")
        return prompt
    
    def rl_eval(self, output_text, criteria=["rules_adherence", "accuracy"]):
        """Evaluate output and provide RL feedback"""
        score = 0
        max_score = len(criteria) * 2
        feedback = []
        
        for criterion in criteria:
            if criterion == "rules_adherence":
                # Check if rules are being followed
                if "TODO.md" in output_text and "branch" in output_text:
                    score += 2
                    feedback.append("✅ Rules adherence: Good")
                else:
                    score += 1
                    feedback.append("⚠️ Rules adherence: Partial")
            
            elif criterion == "accuracy":
                # Check for technical accuracy markers
                if any(word in output_text.lower() for word in ["test", "verify", "check", "validate"]):
                    score += 2
                    feedback.append("✅ Accuracy: Good verification approach")
                else:
                    score += 1
                    feedback.append("⚠️ Accuracy: Could use more validation")
        
        final_score = (score / max_score) * 10
        
        print(f"📊 RL EVALUATION SCORE: {final_score:.1f}/10")
        for item in feedback:
            print(f"  {item}")
        
        if final_score >= 8:
            print("🎉 Excellent performance! Continue optimizing.")
        else:
            print("🔄 Performance needs improvement. Retry with corrections.")
        
        return final_score

def main():
    parser = argparse.ArgumentParser(description="Claude CLI - Rule Enforcement & RL")
    parser.add_argument('command', choices=[
        'rule-check', 'branch-init', 'todo-update', 'env-lock', 
        'sub-agent-spawn', 'rl-eval'
    ])
    parser.add_argument('--task', help='Task description')
    parser.add_argument('--status', help='Task status')
    parser.add_argument('--verify', help='Verification details')
    parser.add_argument('--branch', help='Branch name')
    parser.add_argument('--desc', help='Description')
    parser.add_argument('--key', help='Environment key')
    parser.add_argument('--value', help='Environment value')
    parser.add_argument('--output', default='prompt.xml', help='Output file')
    parser.add_argument('--text', help='Text to evaluate')
    parser.add_argument('--criteria', nargs='+', default=['rules_adherence', 'accuracy'])
    parser.add_argument('--lock', action='store_true', help='Lock environment file')
    
    args = parser.parse_args()
    cli = ClaudeCLI()
    
    if args.command == 'rule-check':
        return 0 if cli.rule_check(args.task) else 1
    elif args.command == 'branch-init':
        return 0 if cli.branch_init(args.branch, args.desc or "") else 1
    elif args.command == 'todo-update':
        cli.todo_update(args.task, args.status, args.verify or "")
    elif args.command == 'env-lock':
        cli.env_lock(args.key, args.value, args.lock)
    elif args.command == 'sub-agent-spawn':
        cli.sub_agent_spawn(args.task, args.output)
    elif args.command == 'rl-eval':
        score = cli.rl_eval(args.text or "", args.criteria)
        return 0 if score >= 8 else 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
