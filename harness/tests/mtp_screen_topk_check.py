"""Fixed draft selector: exact finite membership, valid NaN/tie ranking, replays."""
import argparse
import json
from pathlib import Path
import torch
from vllm.model_executor.models.nemotron_h_mtp_screen import _screen_topk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', required=True)
    args = ap.parse_args()
    torch.manual_seed(197419)
    cases = elements = replays = 0
    for width, keep, offset in [(2048,256,2048),(256,32,0),(2048,32,2048)]:
        for rows in [1,64]:
            x = torch.randn((rows,width),device='cuda',dtype=torch.float32)
            g = torch.cuda.CUDAGraph()
            _screen_topk(x,width,keep,offset)
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                out = _screen_topk(x,width,keep,offset)
            for mode in ['random','all-zero','ties','nan','infinity','random']:
                if mode == 'all-zero': x.zero_(); x[:,::2] = -0.0
                elif mode == 'ties': x.copy_(torch.randint(-8,9,x.shape,device='cuda').float())
                else:
                    x.normal_()
                    if mode == 'nan': x[:,::17] = float('nan')
                    if mode == 'infinity':
                        x[:,::19] = float('inf'); x[:,1::19] = -float('inf')
                g.replay(); replays += 1
                local = out - torch.arange(rows,device='cuda')[:,None] * offset
                assert bool(((local >= 0) & (local < width)).all())
                assert bool((local.sort(-1).values[:,1:] != local.sort(-1).values[:,:-1]).all())
                selected = x.gather(1,local)
                ref = x.topk(keep,-1).values
                # Score multisets must agree even if tied IDs/order differ.
                assert torch.allclose(selected.sort(-1).values,ref.sort(-1).values,rtol=0,atol=0,equal_nan=True), (width,rows,mode)
                if mode == 'random':
                    assert torch.equal(local.sort(-1).values,x.topk(keep,-1).indices.sort(-1).values)
                cases += rows; elements += out.numel()
    result={'all_rankings_valid':True,'random_membership_exact':True,'ties':'lowest-local-ID; can differ from torch.topk','row_cases':cases,'selected_elements':elements,'changed_input_graph_replays':replays}
    Path(args.result).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    print('MTP_SCREEN_TOPK_CHECK_PASS',flush=True)
if __name__ == '__main__': main()
