import os
import typer
from pytimings.timer import function_timer, global_timings, scoped_timing


@function_timer("SE")
def _SE():
    from pymor.discretizers.builtin.grids.tria import TriaGrid
    return TriaGrid((1000, 1000)).subentities(0, 2)


@function_timer("SUE SUI")
def _SUE_SUI(SE):
    from pymor.discretizers.builtin.relations import inverse_relation
    return inverse_relation(SE, with_indices=True)


def benchmark(with_numba: bool=True, repeats: int=10):
    if not with_numba:
        os.environ["NUMBA_DISABLE_JIT"] = "1"
    with scoped_timing("total"):
        from pymor.discretizers.builtin.relations import inverse_relation
        from pymor.discretizers.builtin.grids.tria import TriaGrid
        for i in range(repeats):
            _SUE_SUI(_SE())
    global_timings.add_extra_data({"with_numba": with_numba, "repeats": repeats})
    global_timings.output_console()
    global_timings.output_files(output_dir=".",
                                csv_base=f"benchmark_numba-{with_numba}_repeats-{repeats}")


def run():
    typer.run(benchmark)


if __name__ == '__main__':
    run()
