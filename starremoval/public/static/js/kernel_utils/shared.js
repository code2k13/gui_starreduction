/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
// Import shared functionality from tfjs-backend-cpu without triggering
// side effects.
// tslint:disable-next-line: no-imports-from-dist
import { concatImpl as concatImplCPU } from '@tensorflow/tfjs-backend-cpu/dist/shared';
// tslint:disable-next-line: no-imports-from-dist
import { sliceImpl as sliceImplCPU } from '@tensorflow/tfjs-backend-cpu/dist/shared';
// tslint:disable-next-line: no-imports-from-dist
import { rangeImpl as rangeImplCPU } from '@tensorflow/tfjs-backend-cpu/dist/shared';
// tslint:disable-next-line: no-imports-from-dist
import { stringNGramsImpl as stringNGramsImplCPU } from '@tensorflow/tfjs-backend-cpu/dist/shared';
// tslint:disable-next-line: no-imports-from-dist
import { stringSplitImpl as stringSplitImplCPU } from '@tensorflow/tfjs-backend-cpu/dist/shared';
// tslint:disable-next-line: no-imports-from-dist
import { stringToHashBucketFastImpl as stringToHashBucketFastImplCPU } from '@tensorflow/tfjs-backend-cpu/dist/shared';
export { concatImplCPU, rangeImplCPU, sliceImplCPU, stringNGramsImplCPU, stringSplitImplCPU, stringToHashBucketFastImplCPU };
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic2hhcmVkLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdhc20vc3JjL2tlcm5lbF91dGlscy9zaGFyZWQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsdUVBQXVFO0FBQ3ZFLGdCQUFnQjtBQUNoQixpREFBaUQ7QUFDakQsT0FBTyxFQUFDLFVBQVUsSUFBSSxhQUFhLEVBQUMsTUFBTSwwQ0FBMEMsQ0FBQztBQUNyRixpREFBaUQ7QUFDakQsT0FBTyxFQUFDLFNBQVMsSUFBSSxZQUFZLEVBQUMsTUFBTSwwQ0FBMEMsQ0FBQztBQUNuRixpREFBaUQ7QUFDakQsT0FBTyxFQUFDLFNBQVMsSUFBSSxZQUFZLEVBQUMsTUFBTSwwQ0FBMEMsQ0FBQztBQUNuRixpREFBaUQ7QUFDakQsT0FBTyxFQUFDLGdCQUFnQixJQUFJLG1CQUFtQixFQUFDLE1BQU0sMENBQTBDLENBQUM7QUFDakcsaURBQWlEO0FBQ2pELE9BQU8sRUFBQyxlQUFlLElBQUksa0JBQWtCLEVBQUMsTUFBTSwwQ0FBMEMsQ0FBQztBQUMvRixpREFBaUQ7QUFDakQsT0FBTyxFQUFDLDBCQUEwQixJQUFJLDZCQUE2QixFQUFDLE1BQU0sMENBQTBDLENBQUM7QUFFckgsT0FBTyxFQUNMLGFBQWEsRUFDYixZQUFZLEVBQ1osWUFBWSxFQUNaLG1CQUFtQixFQUNuQixrQkFBa0IsRUFDbEIsNkJBQTZCLEVBQzlCLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbi8vIEltcG9ydCBzaGFyZWQgZnVuY3Rpb25hbGl0eSBmcm9tIHRmanMtYmFja2VuZC1jcHUgd2l0aG91dCB0cmlnZ2VyaW5nXG4vLyBzaWRlIGVmZmVjdHMuXG4vLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLWltcG9ydHMtZnJvbS1kaXN0XG5pbXBvcnQge2NvbmNhdEltcGwgYXMgY29uY2F0SW1wbENQVX0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1iYWNrZW5kLWNwdS9kaXN0L3NoYXJlZCc7XG4vLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLWltcG9ydHMtZnJvbS1kaXN0XG5pbXBvcnQge3NsaWNlSW1wbCBhcyBzbGljZUltcGxDUFV9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtYmFja2VuZC1jcHUvZGlzdC9zaGFyZWQnO1xuLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOiBuby1pbXBvcnRzLWZyb20tZGlzdFxuaW1wb3J0IHtyYW5nZUltcGwgYXMgcmFuZ2VJbXBsQ1BVfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWJhY2tlbmQtY3B1L2Rpc3Qvc2hhcmVkJztcbi8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTogbm8taW1wb3J0cy1mcm9tLWRpc3RcbmltcG9ydCB7c3RyaW5nTkdyYW1zSW1wbCBhcyBzdHJpbmdOR3JhbXNJbXBsQ1BVfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWJhY2tlbmQtY3B1L2Rpc3Qvc2hhcmVkJztcbi8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTogbm8taW1wb3J0cy1mcm9tLWRpc3RcbmltcG9ydCB7c3RyaW5nU3BsaXRJbXBsIGFzIHN0cmluZ1NwbGl0SW1wbENQVX0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1iYWNrZW5kLWNwdS9kaXN0L3NoYXJlZCc7XG4vLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLWltcG9ydHMtZnJvbS1kaXN0XG5pbXBvcnQge3N0cmluZ1RvSGFzaEJ1Y2tldEZhc3RJbXBsIGFzIHN0cmluZ1RvSGFzaEJ1Y2tldEZhc3RJbXBsQ1BVfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWJhY2tlbmQtY3B1L2Rpc3Qvc2hhcmVkJztcblxuZXhwb3J0IHtcbiAgY29uY2F0SW1wbENQVSxcbiAgcmFuZ2VJbXBsQ1BVLFxuICBzbGljZUltcGxDUFUsXG4gIHN0cmluZ05HcmFtc0ltcGxDUFUsXG4gIHN0cmluZ1NwbGl0SW1wbENQVSxcbiAgc3RyaW5nVG9IYXNoQnVja2V0RmFzdEltcGxDUFVcbn07XG4iXX0=