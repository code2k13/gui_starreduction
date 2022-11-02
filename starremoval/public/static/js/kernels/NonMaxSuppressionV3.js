/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
import { NonMaxSuppressionV3 } from '@tensorflow/tfjs-core';
import { parseResultStruct } from './NonMaxSuppression_util';
let wasmFunc;
function setup(backend) {
    wasmFunc = backend.wasm.cwrap(NonMaxSuppressionV3, 'number', // Result*
    [
        'number',
        'number',
        'number',
        'number',
        'number', // scoreThreshold
    ]);
}
function kernelFunc(args) {
    const { backend, inputs, attrs } = args;
    const { iouThreshold, maxOutputSize, scoreThreshold } = attrs;
    const { boxes, scores } = inputs;
    const boxesId = backend.dataIdMap.get(boxes.dataId).id;
    const scoresId = backend.dataIdMap.get(scores.dataId).id;
    const resOffset = wasmFunc(boxesId, scoresId, maxOutputSize, iouThreshold, scoreThreshold);
    const { pSelectedIndices, selectedSize, pSelectedScores, pValidOutputs } = parseResultStruct(backend, resOffset);
    // Since we are not using scores for V3, we have to delete it from the heap.
    backend.wasm._free(pSelectedScores);
    backend.wasm._free(pValidOutputs);
    const selectedIndicesTensor = backend.makeOutput([selectedSize], 'int32', pSelectedIndices);
    return selectedIndicesTensor;
}
export const nonMaxSuppressionV3Config = {
    kernelName: NonMaxSuppressionV3,
    backendName: 'wasm',
    setupFunc: setup,
    kernelFunc: kernelFunc,
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiTm9uTWF4U3VwcHJlc3Npb25WMy5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13YXNtL3NyYy9rZXJuZWxzL05vbk1heFN1cHByZXNzaW9uVjMudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUEyQixtQkFBbUIsRUFBa0UsTUFBTSx1QkFBdUIsQ0FBQztBQUlySixPQUFPLEVBQUMsaUJBQWlCLEVBQUMsTUFBTSwwQkFBMEIsQ0FBQztBQUUzRCxJQUFJLFFBRXVELENBQUM7QUFFNUQsU0FBUyxLQUFLLENBQUMsT0FBb0I7SUFDakMsUUFBUSxHQUFHLE9BQU8sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUN6QixtQkFBbUIsRUFDbkIsUUFBUSxFQUFHLFVBQVU7SUFDckI7UUFDRSxRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUSxFQUFHLGlCQUFpQjtLQUM3QixDQUFDLENBQUM7QUFDVCxDQUFDO0FBRUQsU0FBUyxVQUFVLENBQUMsSUFJbkI7SUFDQyxNQUFNLEVBQUMsT0FBTyxFQUFFLE1BQU0sRUFBRSxLQUFLLEVBQUMsR0FBRyxJQUFJLENBQUM7SUFDdEMsTUFBTSxFQUFDLFlBQVksRUFBRSxhQUFhLEVBQUUsY0FBYyxFQUFDLEdBQUcsS0FBSyxDQUFDO0lBQzVELE1BQU0sRUFBQyxLQUFLLEVBQUUsTUFBTSxFQUFDLEdBQUcsTUFBTSxDQUFDO0lBRS9CLE1BQU0sT0FBTyxHQUFHLE9BQU8sQ0FBQyxTQUFTLENBQUMsR0FBRyxDQUFDLEtBQUssQ0FBQyxNQUFNLENBQUMsQ0FBQyxFQUFFLENBQUM7SUFDdkQsTUFBTSxRQUFRLEdBQUcsT0FBTyxDQUFDLFNBQVMsQ0FBQyxHQUFHLENBQUMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxDQUFDLEVBQUUsQ0FBQztJQUV6RCxNQUFNLFNBQVMsR0FDWCxRQUFRLENBQUMsT0FBTyxFQUFFLFFBQVEsRUFBRSxhQUFhLEVBQUUsWUFBWSxFQUFFLGNBQWMsQ0FBQyxDQUFDO0lBRTdFLE1BQU0sRUFBQyxnQkFBZ0IsRUFBRSxZQUFZLEVBQUUsZUFBZSxFQUFFLGFBQWEsRUFBQyxHQUNsRSxpQkFBaUIsQ0FBQyxPQUFPLEVBQUUsU0FBUyxDQUFDLENBQUM7SUFFMUMsNEVBQTRFO0lBQzVFLE9BQU8sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLGVBQWUsQ0FBQyxDQUFDO0lBQ3BDLE9BQU8sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLGFBQWEsQ0FBQyxDQUFDO0lBRWxDLE1BQU0scUJBQXFCLEdBQ3ZCLE9BQU8sQ0FBQyxVQUFVLENBQUMsQ0FBQyxZQUFZLENBQUMsRUFBRSxPQUFPLEVBQUUsZ0JBQWdCLENBQUMsQ0FBQztJQUVsRSxPQUFPLHFCQUFxQixDQUFDO0FBQy9CLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSx5QkFBeUIsR0FBaUI7SUFDckQsVUFBVSxFQUFFLG1CQUFtQjtJQUMvQixXQUFXLEVBQUUsTUFBTTtJQUNuQixTQUFTLEVBQUUsS0FBSztJQUNoQixVQUFVLEVBQUUsVUFBOEI7Q0FDM0MsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE5IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtLZXJuZWxDb25maWcsIEtlcm5lbEZ1bmMsIE5vbk1heFN1cHByZXNzaW9uVjMsIE5vbk1heFN1cHByZXNzaW9uVjNBdHRycywgTm9uTWF4U3VwcHJlc3Npb25WM0lucHV0cywgVGVuc29ySW5mb30gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtCYWNrZW5kV2FzbX0gZnJvbSAnLi4vYmFja2VuZF93YXNtJztcblxuaW1wb3J0IHtwYXJzZVJlc3VsdFN0cnVjdH0gZnJvbSAnLi9Ob25NYXhTdXBwcmVzc2lvbl91dGlsJztcblxubGV0IHdhc21GdW5jOiAoXG4gICAgYm94ZXNJZDogbnVtYmVyLCBzY29yZXNJZDogbnVtYmVyLCBtYXhPdXRwdXRTaXplOiBudW1iZXIsXG4gICAgaW91VGhyZXNob2xkOiBudW1iZXIsIHNjb3JlVGhyZXNob2xkOiBudW1iZXIpID0+IG51bWJlcjtcblxuZnVuY3Rpb24gc2V0dXAoYmFja2VuZDogQmFja2VuZFdhc20pOiB2b2lkIHtcbiAgd2FzbUZ1bmMgPSBiYWNrZW5kLndhc20uY3dyYXAoXG4gICAgICBOb25NYXhTdXBwcmVzc2lvblYzLFxuICAgICAgJ251bWJlcicsICAvLyBSZXN1bHQqXG4gICAgICBbXG4gICAgICAgICdudW1iZXInLCAgLy8gYm94ZXNJZFxuICAgICAgICAnbnVtYmVyJywgIC8vIHNjb3Jlc0lkXG4gICAgICAgICdudW1iZXInLCAgLy8gbWF4T3V0cHV0U2l6ZVxuICAgICAgICAnbnVtYmVyJywgIC8vIGlvdVRocmVzaG9sZFxuICAgICAgICAnbnVtYmVyJywgIC8vIHNjb3JlVGhyZXNob2xkXG4gICAgICBdKTtcbn1cblxuZnVuY3Rpb24ga2VybmVsRnVuYyhhcmdzOiB7XG4gIGJhY2tlbmQ6IEJhY2tlbmRXYXNtLFxuICBpbnB1dHM6IE5vbk1heFN1cHByZXNzaW9uVjNJbnB1dHMsXG4gIGF0dHJzOiBOb25NYXhTdXBwcmVzc2lvblYzQXR0cnNcbn0pOiBUZW5zb3JJbmZvIHtcbiAgY29uc3Qge2JhY2tlbmQsIGlucHV0cywgYXR0cnN9ID0gYXJncztcbiAgY29uc3Qge2lvdVRocmVzaG9sZCwgbWF4T3V0cHV0U2l6ZSwgc2NvcmVUaHJlc2hvbGR9ID0gYXR0cnM7XG4gIGNvbnN0IHtib3hlcywgc2NvcmVzfSA9IGlucHV0cztcblxuICBjb25zdCBib3hlc0lkID0gYmFja2VuZC5kYXRhSWRNYXAuZ2V0KGJveGVzLmRhdGFJZCkuaWQ7XG4gIGNvbnN0IHNjb3Jlc0lkID0gYmFja2VuZC5kYXRhSWRNYXAuZ2V0KHNjb3Jlcy5kYXRhSWQpLmlkO1xuXG4gIGNvbnN0IHJlc09mZnNldCA9XG4gICAgICB3YXNtRnVuYyhib3hlc0lkLCBzY29yZXNJZCwgbWF4T3V0cHV0U2l6ZSwgaW91VGhyZXNob2xkLCBzY29yZVRocmVzaG9sZCk7XG5cbiAgY29uc3Qge3BTZWxlY3RlZEluZGljZXMsIHNlbGVjdGVkU2l6ZSwgcFNlbGVjdGVkU2NvcmVzLCBwVmFsaWRPdXRwdXRzfSA9XG4gICAgICBwYXJzZVJlc3VsdFN0cnVjdChiYWNrZW5kLCByZXNPZmZzZXQpO1xuXG4gIC8vIFNpbmNlIHdlIGFyZSBub3QgdXNpbmcgc2NvcmVzIGZvciBWMywgd2UgaGF2ZSB0byBkZWxldGUgaXQgZnJvbSB0aGUgaGVhcC5cbiAgYmFja2VuZC53YXNtLl9mcmVlKHBTZWxlY3RlZFNjb3Jlcyk7XG4gIGJhY2tlbmQud2FzbS5fZnJlZShwVmFsaWRPdXRwdXRzKTtcblxuICBjb25zdCBzZWxlY3RlZEluZGljZXNUZW5zb3IgPVxuICAgICAgYmFja2VuZC5tYWtlT3V0cHV0KFtzZWxlY3RlZFNpemVdLCAnaW50MzInLCBwU2VsZWN0ZWRJbmRpY2VzKTtcblxuICByZXR1cm4gc2VsZWN0ZWRJbmRpY2VzVGVuc29yO1xufVxuXG5leHBvcnQgY29uc3Qgbm9uTWF4U3VwcHJlc3Npb25WM0NvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBOb25NYXhTdXBwcmVzc2lvblYzLFxuICBiYWNrZW5kTmFtZTogJ3dhc20nLFxuICBzZXR1cEZ1bmM6IHNldHVwLFxuICBrZXJuZWxGdW5jOiBrZXJuZWxGdW5jIGFzIHt9IGFzIEtlcm5lbEZ1bmMsXG59O1xuIl19