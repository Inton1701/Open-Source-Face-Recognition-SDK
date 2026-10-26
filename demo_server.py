# Local web demo: compare two photos (upload or camera) in the browser.
# Usage: python demo_server.py [port]   then open http://127.0.0.1:8765
import base64, json, os, sys, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import cv2
import numpy as np
from run import GetImageInfo, get_similarity, threshold, tuned_threshold

here = os.path.dirname(os.path.abspath(__file__))
MAX_BODY = 25 * 2**20

def decode(data_url):
    raw = base64.b64decode(data_url.split(',', 1)[-1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

def png(img):
    return 'data:image/png;base64,' + base64.b64encode(cv2.imencode('.png', img)[1]).decode()

def analyse(img, tuned):
    n, boxes, scores, _, aligns, feats = GetImageInfo(img, 10, tuned=tuned)
    faces = [{'box': [round(float(v)) for v in b], 'confidence': round(float(s), 3)} for b, s in zip(boxes, scores)]
    if n == 0:
        return None, faces, None
    # compare the largest face in each photo
    i = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]))
    faces[i]['used'] = True
    return feats[i], faces, png(aligns[i])

class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, ctype='application/json'):
        body = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            with open(os.path.join(here, 'demo.html'), 'rb') as f:
                return self.send(200, f.read(), 'text/html; charset=utf-8')
        samples = {'/sample/1': ('1.jpg', 'image/jpeg'), '/sample/2': ('2.png', 'image/png')}
        if self.path in samples:
            name, ctype = samples[self.path]
            with open(os.path.join(here, 'test', name), 'rb') as f:
                return self.send(200, f.read(), ctype)
        self.send(404, {'error': 'not found'})

    def do_POST(self):
        if self.path != '/compare':
            return self.send(404, {'error': 'not found'})
        size = int(self.headers.get('Content-Length', 0))
        if size > MAX_BODY:
            return self.send(413, {'error': 'Photos are too large; keep both under 25 MB together.'})
        try:
            req = json.loads(self.rfile.read(size))
            tuned = bool(req.get('tuned'))
            t = time.perf_counter()
            out = []
            for key in ('img1', 'img2'):
                img = decode(req[key])
                if img is None:
                    raise ValueError('Photo %s is not a readable image.' % key[-1])
                out.append(analyse(img, tuned))
            limit = tuned_threshold if tuned else threshold
            result = {'faces': [o[1] for o in out], 'crops': [o[2] for o in out], 'threshold': round(limit, 1),
                      'ms': round((time.perf_counter() - t) * 1000)}
            if out[0][0] is not None and out[1][0] is not None:
                score = float(get_similarity(out[0][0], out[1][0]))
                result.update(score=round(score, 2), same=score > limit)
            self.send(200, result)
        except (ValueError, TypeError, KeyError) as e:
            self.send(400, {'error': str(e)})

    def log_message(self, fmt, *args):
        pass

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    host = sys.argv[2] if len(sys.argv) > 2 else '127.0.0.1'  # 0.0.0.0 inside a container
    print('Face compare demo on http://%s:%d' % (host, port), flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()
