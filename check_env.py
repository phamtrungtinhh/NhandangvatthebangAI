import sys
def main():
    try:
        import ultralytics
        print('ultralytics', getattr(ultralytics, '__version__', 'unknown'))
    except Exception as e:
        print('ultralytics import failed:', e)
    try:
        import torch
        print('torch', torch.__version__, 'cuda_available=', torch.cuda.is_available())
    except Exception as e:
        print('torch import failed:', e)

if __name__ == '__main__':
    main()
