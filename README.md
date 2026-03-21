Link sever app : Nhận diện vật thể bằng AI : https://huggingface.co/spaces/phamtrungtinh/smartfocus-ui
# NhandangvatthebangAI

Ung dung Streamlit nhan dang va dem `hoa`, `trai cay`, `cay` va `ta` tu anh/video bang YOLO.

Repository nay da duoc dong goi de nguoi khac co the clone va chay ngay bang Docker, khong can thiet lap moi truong Python thu cong.

## Chay nhanh bang Docker

### Cach 1: Docker Compose

```bash
docker compose up --build
```

Sau khi container len, mo:

```text
http://localhost:8501
```

### Cach 2: Docker CLI

```bash
docker build -t smartfocus-ai .
docker run --rm -p 8501:7860 smartfocus-ai
```

Sau khi container len, mo:

```text
http://localhost:8501
```

## File quan trong

- `app.py`: giao dien va logic xu ly chinh
- `Dockerfile`: dong goi ung dung
- `compose.yaml`: chay nhanh bang Docker Compose
- `.streamlit/config.toml`: cau hinh Streamlit cho local va proxy
- `model_custom/weights/custo_all.pt`: model custom
- `models/flower_best.pt`: model hoa bo tro
- `yolo11n.pt`: model COCO
- `analysis.db`: du lieu lich su phan tich

## Ghi chu

- Cac thu muc tam, output thu nghiem, moi truong ao, va du lieu bao cao da duoc loai khoi luong dong goi.
- Thu muc `BAOCAODOAN2` khong nam trong lan cap nhat dong goi nay.
- Neu muon chay khong dung Docker, ban co the cai `requirements.txt` va chay `streamlit run app.py`.

## Yeu cau

- Docker Desktop hoac Docker Engine
- Toi thieu RAM nen tu `4 GB` tro len de tai model va xu ly anh on dinh
