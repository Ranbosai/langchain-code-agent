import click
from agent import create_agent

@click.group()
@click.pass_context
def cli(ctx):
    """Command-line interface for code writer and explainer."""
    ctx.ensure_object(dict)

@cli.command()
@click.option("--language", required=True, help="Programming language (e.g. python, javascript).")
@click.option("--task", required=True, help="Task description to generate code for.")
def write(language, task):
    """Generate code for a given task in the target language."""
    agent = create_agent()
    user_prompt = f"Write {language} code to: {task}"
    response = agent.run(user_prompt)
    click.echo(response)

@cli.command()
@click.option("--language", required=True, help="Programming language of the code.")
@click.option("--code", required=True, help="Code snippet to explain.")
def explain(language, code):
    """Explain what the provided code does."""
    agent = create_agent()
    user_prompt = f"Explain the following {language} code:\n{code}"
    response = agent.run(user_prompt)
    click.echo(response)

if __name__ == "__main__":
    cli()
