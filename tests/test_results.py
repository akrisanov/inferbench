from pathlib import Path

from inferbench.results import load_vllm_result


def test_load_vllm_result() -> None:
    path = Path(__file__).parent / "fixtures" / "vllm-result.json"

    result = load_vllm_result(path)

    assert result.duration == 100.5
    assert result.completed == 100
    assert result.failed == 0
    assert result.total_input_tokens == 204800
    assert result.total_output_tokens == 25600
    assert result.request_throughput == 0.995
    assert result.output_throughput == 254.73
    assert result.total_token_throughput == 2292.54

    assert result.mean_ttft_ms == 42.3
    assert result.mean_tpot_ms == 12.1
    assert result.mean_itl_ms == 12.0
    assert result.mean_e2el_ms == 3128.4

    assert result.percentiles.ttft_ms == {99: 91.4}
    assert result.percentiles.tpot_ms == {99: 17.2}
    assert result.percentiles.itl_ms == {99: 18.1}
    assert result.percentiles.e2el_ms == {99: 4280.5}
