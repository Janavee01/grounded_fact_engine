import requests
import json

# Test PDF upload
def test_upload():
    with open('sample.pdf', 'rb') as f:
        files = {'file': ('sample.pdf', f, 'application/pdf')}
        response = requests.post('http://localhost:8000/upload', files=files)
    print(response.json())

# Test get facts
def test_get_facts():
    response = requests.get('http://localhost:8000/facts')
    print(json.dumps(response.json(), indent=2))

# Test compare
def test_compare():
    response = requests.get('http://localhost:8000/compare')
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_upload()
    test_get_facts()
    test_compare()