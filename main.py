import argparse
import textwrap
from store_agent import run_agent
from erc3 import ERC3

# pip install python-dotenv
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


def parse_args():
    parser = argparse.ArgumentParser(
        description="ERC3 Store Agent - LangChain implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # Run all tasks with OpenRouter
  python main.py -t 3                   # Run only task #3
  python main.py -t 1-5                 # Run tasks 1 through 5
  python main.py -s                     # Stop on first failed task (score=0)
  python main.py -l                     # List all tasks without running
  python main.py -t 5 -s                # Run from task #5, stop on fail
  python main.py -p gigachat -m GigaChat-Pro  # Use GigaChat
"""
    )
    parser.add_argument(
        "-t", "--task",
        type=str,
        default=None,
        help="Task number to run (e.g., '3') or range (e.g., '1-5')"
    )
    parser.add_argument(
        "-s", "--stop-on-fail",
        action="store_true",
        help="Stop after first task with score=0"
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List all tasks without running them"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Model ID to use (default depends on provider)"
    )
    parser.add_argument(
        "-p", "--provider",
        type=str,
        choices=["openrouter", "gigachat"],
        default="openrouter",
        help="LLM provider: openrouter or gigachat (default: openrouter)"
    )
    return parser.parse_args()


def parse_task_range(task_arg: str, max_tasks: int) -> tuple[int, int]:
    """Parse task argument into (start, end) indices (1-based)."""
    if "-" in task_arg:
        parts = task_arg.split("-")
        start = int(parts[0])
        end = int(parts[1])
    else:
        start = int(task_arg)
        end = start
    
    # Validate range
    if start < 1 or end > max_tasks or start > end:
        raise ValueError(f"Invalid task range: {task_arg}. Valid range: 1-{max_tasks}")
    
    return start, end


def main():
    args = parse_args()
    
    core = ERC3()
    provider = args.provider
    
    # Set default model based on provider
    if args.model:
        model_id = args.model
    elif provider == "gigachat":
        model_id = "GigaChat-2-Max"
    else:
        model_id = "openai/gpt-4o"

    # Start session with metadata
    res = core.start_session(
        benchmark="store",
        workspace="my",
        name=f"LangChain Agent ({model_id})",
        architecture=f"LangChain create_agent with {provider}",
        flags=["compete_accuracy"]
    )
    
    print(f"Provider: {provider}, Model: {model_id}")

    status = core.session_status(res.session_id)
    tasks = status.tasks
    print(f"Session has {len(tasks)} tasks")

    # List mode - just show tasks and exit
    if args.list:
        print("\nAvailable tasks:")
        for i, task in enumerate(tasks, 1):
            print(f"  {i:2d}. [{task.spec_id}] {task.task_text}")
        return

    # Determine which tasks to run
    run_all = args.task is None
    if args.task:
        start_idx, end_idx = parse_task_range(args.task, len(tasks))
        tasks_to_run = [(i, tasks[i-1]) for i in range(start_idx, end_idx + 1)]
        print(f"Running tasks {start_idx}-{end_idx}")
    else:
        tasks_to_run = [(i+1, task) for i, task in enumerate(tasks)]

    # Run tasks
    completed_count = 0
    stopped_early = False
    
    for task_num, task in tasks_to_run:
        print("=" * 40)
        print(f"Task #{task_num}: {task.task_id} ({task.spec_id})")
        print(f"  {task.task_text}")
        
        core.start_task(task)
        try:
            run_agent(model_id, core, task, provider=provider)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        
        result = core.complete_task(task)
        completed_count += 1
        
        score = 0
        if result.eval:
            score = result.eval.score
            explain = textwrap.indent(result.eval.logs, "  ")
            print(f"\nSCORE: {score}\n{explain}\n")
        
        # Stop on fail if requested
        if args.stop_on_fail and score == 0:
            print(f"\n⛔ Stopping: task #{task_num} failed (score=0)")
            stopped_early = True
            break

    # Only submit session if all tasks were run
    if run_all and not stopped_early:
        core.submit_session(res.session_id)
        print(f"\n✅ Session submitted: {res.session_id}")
    else:
        print(f"\n📋 Session NOT submitted (ran {completed_count}/{len(tasks)} tasks)")
        print(f"   Session ID: {res.session_id}")


if __name__ == "__main__":
    main()











