import sys, getopt, click

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group(
    help=f"\n  Run SELSP optimizer by cmd\n",
    epilog="Try 'python run.py train' to start the default training",
    context_settings=CONTEXT_SETTINGS,
)
@click.pass_context
def cli(ctx: click.Context):
    pass


@cli.command("train", short_help="Run training program")
@click.option("--net_type", "-t", default='ssa', help="Select the type of network to use", type=click.Choice(['ssa', 'mlp']), show_default=False)
@click.option("--env_no", "-e", default=3, help="Set the number [i] of simulation environment, where i means there are i products", type=click.Choice(['3', '4', '5', '6']), show_default=False)
@click.pass_context
def train(ctx: click.Context, net_type: str, env_no: str):
    if net_type == 'ssa':
        from experiment.standard_self_attention.run import train_ssa
        train_ssa(int(env_no))
    elif net_type == 'mlp':
        from experiment.mlp.run import train_mlp
        train_mlp(int(env_no))


@cli.command("evaluate", short_help="Run evaluating program")
@click.option("--net_type", "-t", default='ssa', help="Select the type of network to use", type=click.Choice(['ssa', 'mlp']), show_default=False)
@click.option("--env_no", "-e", default='3', help="Set the number [i] of simulation environment, where i means there are i products", type=click.Choice(['3', '4', '5', '6']), show_default=False)
@click.option("--demand_scale", "-s", default=1, help="Set the demand scale of all products", type=float, show_default=False)
@click.option("--model_path", "-m", default='./files/ssa-model3', help="Give the path of a trainded model file to evaluate", type=str, show_default=False)
@click.option("--result_xlsx_path", "-r", default='./files/evaluate_result.xlsx', help="Give the path of the result xlsx file", type=str, show_default=False)
@click.pass_context
def train(ctx: click.Context, net_type: str, env_no: str, demand_scale: float, model_path: str, result_xlsx_path: str):
    if net_type == 'ssa':
        from experiment.standard_self_attention.evaluate_elsp import evaluate_ssa
        evaluate_ssa(int(env_no), demand_scale, model_path, result_xlsx_path)
    elif net_type == 'mlp':
        from experiment.mlp.evaluate_elsp import evaluate_mlp
        evaluate_mlp(int(env_no), demand_scale, model_path, result_xlsx_path)


def main():
    cli()


if __name__ == '__main__':
    main()