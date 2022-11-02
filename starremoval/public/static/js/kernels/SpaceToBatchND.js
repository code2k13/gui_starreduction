/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import { backend_util, SpaceToBatchND, util } from '@tensorflow/tfjs-core';
import { padV2Config } from './PadV2';
import { reshape } from './Reshape';
import { transpose } from './Transpose';
function spaceToBatchND(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { blockShape, paddings } = attrs;
    const prod = util.sizeFromShape(blockShape);
    const completePaddings = [[0, 0]];
    completePaddings.push(...paddings);
    for (let i = 1 + blockShape.length; i < x.shape.length; ++i) {
        completePaddings.push([0, 0]);
    }
    const paddedX = padV2Config.kernelFunc({
        inputs: { x },
        backend,
        attrs: { paddings: completePaddings, constantValue: 0 }
    });
    const reshapedPaddedShape = backend_util.getReshaped(paddedX.shape, blockShape, prod, false);
    const permutedReshapedPaddedPermutation = backend_util.getPermuted(reshapedPaddedShape.length, blockShape.length, false);
    const flattenShape = backend_util.getReshapedPermuted(paddedX.shape, blockShape, prod, false);
    const reshapeInputs = { x: paddedX };
    const reshapeAttrs = { shape: reshapedPaddedShape };
    const paddedXReshaped = reshape({ inputs: reshapeInputs, backend, attrs: reshapeAttrs });
    const transposeInputs = { x: paddedXReshaped };
    const transposeAttrs = { perm: permutedReshapedPaddedPermutation };
    const paddedXT = transpose({ inputs: transposeInputs, backend, attrs: transposeAttrs });
    const resultReshapeInputs = { x: paddedXT };
    const resultReshapeAttrs = { shape: flattenShape };
    const result = reshape({ inputs: resultReshapeInputs, backend, attrs: resultReshapeAttrs });
    backend.disposeData(paddedX.dataId);
    backend.disposeData(paddedXReshaped.dataId);
    backend.disposeData(paddedXT.dataId);
    return result;
}
export const spaceToBatchNDConfig = {
    kernelName: SpaceToBatchND,
    backendName: 'wasm',
    kernelFunc: spaceToBatchND
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiU3BhY2VUb0JhdGNoTkQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2FzbS9zcmMva2VybmVscy9TcGFjZVRvQmF0Y2hORC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsWUFBWSxFQUF5RCxjQUFjLEVBQTBGLElBQUksRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBSXhOLE9BQU8sRUFBQyxXQUFXLEVBQUMsTUFBTSxTQUFTLENBQUM7QUFDcEMsT0FBTyxFQUFDLE9BQU8sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNsQyxPQUFPLEVBQUMsU0FBUyxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRXRDLFNBQVMsY0FBYyxDQUFDLElBSXZCO0lBQ0MsTUFBTSxFQUFDLE1BQU0sRUFBRSxPQUFPLEVBQUUsS0FBSyxFQUFDLEdBQUcsSUFBSSxDQUFDO0lBQ3RDLE1BQU0sRUFBQyxDQUFDLEVBQUMsR0FBRyxNQUFNLENBQUM7SUFDbkIsTUFBTSxFQUFDLFVBQVUsRUFBRSxRQUFRLEVBQUMsR0FBRyxLQUFLLENBQUM7SUFFckMsTUFBTSxJQUFJLEdBQUcsSUFBSSxDQUFDLGFBQWEsQ0FBQyxVQUFVLENBQUMsQ0FBQztJQUU1QyxNQUFNLGdCQUFnQixHQUE0QixDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDM0QsZ0JBQWdCLENBQUMsSUFBSSxDQUFDLEdBQUksUUFBb0MsQ0FBQyxDQUFDO0lBRWhFLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxHQUFHLFVBQVUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxHQUFHLENBQUMsQ0FBQyxLQUFLLENBQUMsTUFBTSxFQUFFLEVBQUUsQ0FBQyxFQUFFO1FBQzNELGdCQUFnQixDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0tBQy9CO0lBRUQsTUFBTSxPQUFPLEdBQUcsV0FBVyxDQUFDLFVBQVUsQ0FBQztRQUNyQyxNQUFNLEVBQUUsRUFBQyxDQUFDLEVBQUM7UUFDWCxPQUFPO1FBQ1AsS0FBSyxFQUFFLEVBQUMsUUFBUSxFQUFFLGdCQUFnQixFQUFFLGFBQWEsRUFBRSxDQUFDLEVBQUM7S0FDdEQsQ0FBZSxDQUFDO0lBRWpCLE1BQU0sbUJBQW1CLEdBQ3JCLFlBQVksQ0FBQyxXQUFXLENBQUMsT0FBTyxDQUFDLEtBQUssRUFBRSxVQUFVLEVBQUUsSUFBSSxFQUFFLEtBQUssQ0FBQyxDQUFDO0lBRXJFLE1BQU0saUNBQWlDLEdBQUcsWUFBWSxDQUFDLFdBQVcsQ0FDOUQsbUJBQW1CLENBQUMsTUFBTSxFQUFFLFVBQVUsQ0FBQyxNQUFNLEVBQUUsS0FBSyxDQUFDLENBQUM7SUFFMUQsTUFBTSxZQUFZLEdBQ2QsWUFBWSxDQUFDLG1CQUFtQixDQUFDLE9BQU8sQ0FBQyxLQUFLLEVBQUUsVUFBVSxFQUFFLElBQUksRUFBRSxLQUFLLENBQUMsQ0FBQztJQUU3RSxNQUFNLGFBQWEsR0FBa0IsRUFBQyxDQUFDLEVBQUUsT0FBTyxFQUFDLENBQUM7SUFDbEQsTUFBTSxZQUFZLEdBQWlCLEVBQUMsS0FBSyxFQUFFLG1CQUFtQixFQUFDLENBQUM7SUFDaEUsTUFBTSxlQUFlLEdBQ2pCLE9BQU8sQ0FBQyxFQUFDLE1BQU0sRUFBRSxhQUFhLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBRSxZQUFZLEVBQUMsQ0FBQyxDQUFDO0lBRW5FLE1BQU0sZUFBZSxHQUFvQixFQUFDLENBQUMsRUFBRSxlQUFlLEVBQUMsQ0FBQztJQUM5RCxNQUFNLGNBQWMsR0FDQyxFQUFDLElBQUksRUFBRSxpQ0FBaUMsRUFBQyxDQUFDO0lBQy9ELE1BQU0sUUFBUSxHQUNWLFNBQVMsQ0FBQyxFQUFDLE1BQU0sRUFBRSxlQUFlLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBRSxjQUFjLEVBQUMsQ0FBQyxDQUFDO0lBRXpFLE1BQU0sbUJBQW1CLEdBQWtCLEVBQUMsQ0FBQyxFQUFFLFFBQVEsRUFBQyxDQUFDO0lBQ3pELE1BQU0sa0JBQWtCLEdBQWlCLEVBQUMsS0FBSyxFQUFFLFlBQVksRUFBQyxDQUFDO0lBQy9ELE1BQU0sTUFBTSxHQUFHLE9BQU8sQ0FDbEIsRUFBQyxNQUFNLEVBQUUsbUJBQW1CLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBRSxrQkFBa0IsRUFBQyxDQUFDLENBQUM7SUFFdkUsT0FBTyxDQUFDLFdBQVcsQ0FBQyxPQUFPLENBQUMsTUFBTSxDQUFDLENBQUM7SUFDcEMsT0FBTyxDQUFDLFdBQVcsQ0FBQyxlQUFlLENBQUMsTUFBTSxDQUFDLENBQUM7SUFDNUMsT0FBTyxDQUFDLFdBQVcsQ0FBQyxRQUFRLENBQUMsTUFBTSxDQUFDLENBQUM7SUFFckMsT0FBTyxNQUFNLENBQUM7QUFDaEIsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLG9CQUFvQixHQUFpQjtJQUNoRCxVQUFVLEVBQUUsY0FBYztJQUMxQixXQUFXLEVBQUUsTUFBTTtJQUNuQixVQUFVLEVBQUUsY0FBa0M7Q0FDL0MsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIxIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtiYWNrZW5kX3V0aWwsIEtlcm5lbENvbmZpZywgS2VybmVsRnVuYywgUmVzaGFwZUF0dHJzLCBSZXNoYXBlSW5wdXRzLCBTcGFjZVRvQmF0Y2hORCwgU3BhY2VUb0JhdGNoTkRBdHRycywgU3BhY2VUb0JhdGNoTkRJbnB1dHMsIFRlbnNvckluZm8sIFRyYW5zcG9zZUF0dHJzLCBUcmFuc3Bvc2VJbnB1dHMsIHV0aWx9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7QmFja2VuZFdhc219IGZyb20gJy4uL2JhY2tlbmRfd2FzbSc7XG5cbmltcG9ydCB7cGFkVjJDb25maWd9IGZyb20gJy4vUGFkVjInO1xuaW1wb3J0IHtyZXNoYXBlfSBmcm9tICcuL1Jlc2hhcGUnO1xuaW1wb3J0IHt0cmFuc3Bvc2V9IGZyb20gJy4vVHJhbnNwb3NlJztcblxuZnVuY3Rpb24gc3BhY2VUb0JhdGNoTkQoYXJnczoge1xuICBpbnB1dHM6IFNwYWNlVG9CYXRjaE5ESW5wdXRzLFxuICBiYWNrZW5kOiBCYWNrZW5kV2FzbSxcbiAgYXR0cnM6IFNwYWNlVG9CYXRjaE5EQXR0cnNcbn0pIHtcbiAgY29uc3Qge2lucHV0cywgYmFja2VuZCwgYXR0cnN9ID0gYXJncztcbiAgY29uc3Qge3h9ID0gaW5wdXRzO1xuICBjb25zdCB7YmxvY2tTaGFwZSwgcGFkZGluZ3N9ID0gYXR0cnM7XG5cbiAgY29uc3QgcHJvZCA9IHV0aWwuc2l6ZUZyb21TaGFwZShibG9ja1NoYXBlKTtcblxuICBjb25zdCBjb21wbGV0ZVBhZGRpbmdzOiBBcnJheTxbbnVtYmVyLCBudW1iZXJdPiA9IFtbMCwgMF1dO1xuICBjb21wbGV0ZVBhZGRpbmdzLnB1c2goLi4uKHBhZGRpbmdzIGFzIEFycmF5PFtudW1iZXIsIG51bWJlcl0+KSk7XG5cbiAgZm9yIChsZXQgaSA9IDEgKyBibG9ja1NoYXBlLmxlbmd0aDsgaSA8IHguc2hhcGUubGVuZ3RoOyArK2kpIHtcbiAgICBjb21wbGV0ZVBhZGRpbmdzLnB1c2goWzAsIDBdKTtcbiAgfVxuXG4gIGNvbnN0IHBhZGRlZFggPSBwYWRWMkNvbmZpZy5rZXJuZWxGdW5jKHtcbiAgICBpbnB1dHM6IHt4fSxcbiAgICBiYWNrZW5kLFxuICAgIGF0dHJzOiB7cGFkZGluZ3M6IGNvbXBsZXRlUGFkZGluZ3MsIGNvbnN0YW50VmFsdWU6IDB9XG4gIH0pIGFzIFRlbnNvckluZm87XG5cbiAgY29uc3QgcmVzaGFwZWRQYWRkZWRTaGFwZSA9XG4gICAgICBiYWNrZW5kX3V0aWwuZ2V0UmVzaGFwZWQocGFkZGVkWC5zaGFwZSwgYmxvY2tTaGFwZSwgcHJvZCwgZmFsc2UpO1xuXG4gIGNvbnN0IHBlcm11dGVkUmVzaGFwZWRQYWRkZWRQZXJtdXRhdGlvbiA9IGJhY2tlbmRfdXRpbC5nZXRQZXJtdXRlZChcbiAgICAgIHJlc2hhcGVkUGFkZGVkU2hhcGUubGVuZ3RoLCBibG9ja1NoYXBlLmxlbmd0aCwgZmFsc2UpO1xuXG4gIGNvbnN0IGZsYXR0ZW5TaGFwZSA9XG4gICAgICBiYWNrZW5kX3V0aWwuZ2V0UmVzaGFwZWRQZXJtdXRlZChwYWRkZWRYLnNoYXBlLCBibG9ja1NoYXBlLCBwcm9kLCBmYWxzZSk7XG5cbiAgY29uc3QgcmVzaGFwZUlucHV0czogUmVzaGFwZUlucHV0cyA9IHt4OiBwYWRkZWRYfTtcbiAgY29uc3QgcmVzaGFwZUF0dHJzOiBSZXNoYXBlQXR0cnMgPSB7c2hhcGU6IHJlc2hhcGVkUGFkZGVkU2hhcGV9O1xuICBjb25zdCBwYWRkZWRYUmVzaGFwZWQgPVxuICAgICAgcmVzaGFwZSh7aW5wdXRzOiByZXNoYXBlSW5wdXRzLCBiYWNrZW5kLCBhdHRyczogcmVzaGFwZUF0dHJzfSk7XG5cbiAgY29uc3QgdHJhbnNwb3NlSW5wdXRzOiBUcmFuc3Bvc2VJbnB1dHMgPSB7eDogcGFkZGVkWFJlc2hhcGVkfTtcbiAgY29uc3QgdHJhbnNwb3NlQXR0cnM6XG4gICAgICBUcmFuc3Bvc2VBdHRycyA9IHtwZXJtOiBwZXJtdXRlZFJlc2hhcGVkUGFkZGVkUGVybXV0YXRpb259O1xuICBjb25zdCBwYWRkZWRYVCA9XG4gICAgICB0cmFuc3Bvc2Uoe2lucHV0czogdHJhbnNwb3NlSW5wdXRzLCBiYWNrZW5kLCBhdHRyczogdHJhbnNwb3NlQXR0cnN9KTtcblxuICBjb25zdCByZXN1bHRSZXNoYXBlSW5wdXRzOiBSZXNoYXBlSW5wdXRzID0ge3g6IHBhZGRlZFhUfTtcbiAgY29uc3QgcmVzdWx0UmVzaGFwZUF0dHJzOiBSZXNoYXBlQXR0cnMgPSB7c2hhcGU6IGZsYXR0ZW5TaGFwZX07XG4gIGNvbnN0IHJlc3VsdCA9IHJlc2hhcGUoXG4gICAgICB7aW5wdXRzOiByZXN1bHRSZXNoYXBlSW5wdXRzLCBiYWNrZW5kLCBhdHRyczogcmVzdWx0UmVzaGFwZUF0dHJzfSk7XG5cbiAgYmFja2VuZC5kaXNwb3NlRGF0YShwYWRkZWRYLmRhdGFJZCk7XG4gIGJhY2tlbmQuZGlzcG9zZURhdGEocGFkZGVkWFJlc2hhcGVkLmRhdGFJZCk7XG4gIGJhY2tlbmQuZGlzcG9zZURhdGEocGFkZGVkWFQuZGF0YUlkKTtcblxuICByZXR1cm4gcmVzdWx0O1xufVxuXG5leHBvcnQgY29uc3Qgc3BhY2VUb0JhdGNoTkRDb25maWc6IEtlcm5lbENvbmZpZyA9IHtcbiAga2VybmVsTmFtZTogU3BhY2VUb0JhdGNoTkQsXG4gIGJhY2tlbmROYW1lOiAnd2FzbScsXG4gIGtlcm5lbEZ1bmM6IHNwYWNlVG9CYXRjaE5EIGFzIHt9IGFzIEtlcm5lbEZ1bmNcbn07XG4iXX0=