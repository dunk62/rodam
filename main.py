"""
카카오톡 챗봇 스킬 서버

Google Drive를 지식 베이스로 사용:
- CSV 파일로 재고 확인 (최신 파일 자동 감지, 10분 캐싱)
- PDF 파일로 제품 매뉴얼 답변 (Gemini API 활용)
"""

import os
import io
import json
import tempfile
from typing import Optional, Dict, Any
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from cachetools import TTLCache

# Google APIs
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import google.generativeai as genai

# 환경변수 로드
load_dotenv()

# 설정
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "182hEKlJPGxDBmOKspNNS3uunUdF7bRv7")
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "service-account-key.json")
SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")  # Railway용 JSON 문자열

# Gemini API 설정
genai.configure(api_key=GOOGLE_API_KEY)

# Google Drive API 설정
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# FastAPI 앱 생성
app = FastAPI(
    title="카카오톡 챗봇 스킬 서버",
    description="Google Drive 기반 재고 확인 + PDF 매뉴얼 답변",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ 카카오 요청/응답 모델 ============

class KakaoUser(BaseModel):
    id: str
    type: str = "botUserKey"
    properties: Dict[str, Any] = {}

class KakaoUserRequest(BaseModel):
    timezone: str = "Asia/Seoul"
    block: Dict[str, Any] = {}
    utterance: str
    lang: str = "ko"
    user: KakaoUser

class KakaoBot(BaseModel):
    id: str
    name: str

class KakaoIntent(BaseModel):
    id: str
    name: str

class KakaoAction(BaseModel):
    id: str
    name: str
    params: Dict[str, str] = {}
    detailParams: Dict[str, Any] = {}
    clientExtra: Dict[str, Any] = {}

class KakaoRequest(BaseModel):
    intent: KakaoIntent
    userRequest: KakaoUserRequest
    bot: KakaoBot
    action: KakaoAction


# 캐시 설정 (10분 TTL)
inventory_cache = TTLCache(maxsize=1, ttl=600)
pdf_cache = TTLCache(maxsize=10, ttl=600)


def get_drive_service():
    """Google Drive 서비스 객체 생성"""
    try:
        # Railway 배포: 환경변수에서 JSON 직접 읽기
        if SERVICE_ACCOUNT_JSON:
            service_info = json.loads(SERVICE_ACCOUNT_JSON)
            credentials = service_account.Credentials.from_service_account_info(
                service_info,
                scopes=SCOPES
            )
        # 로컬 개발: 파일에서 읽기
        else:
            credentials = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE,
                scopes=SCOPES
            )
        return build('drive', 'v3', credentials=credentials)
    except Exception as e:
        print(f"Google Drive 인증 실패: {str(e)}")
        return None


def search_files_in_folder(mime_type: str, order_by: str = None) -> list:
    """타겟 폴더 내에서 특정 MIME 타입의 파일 검색"""
    drive_service = get_drive_service()
    if not drive_service:
        return []
    
    query = f"'{FOLDER_ID}' in parents and mimeType='{mime_type}' and trashed=false"
    
    params = {
        'q': query,
        'fields': 'files(id, name, createdTime, mimeType)',
        'pageSize': 100
    }
    
    if order_by:
        params['orderBy'] = order_by
    
    try:
        results = drive_service.files().list(**params).execute()
        return results.get('files', [])
    except Exception as e:
        print(f"파일 검색 실패: {str(e)}")
        return []


def download_file_content(file_id: str) -> bytes:
    """Google Drive에서 파일 콘텐츠 다운로드"""
    drive_service = get_drive_service()
    if not drive_service:
        return b''
    
    try:
        request = drive_service.files().get_media(fileId=file_id)
        file_buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(file_buffer, request)
        
        done = False
        while not done:
            _, done = downloader.next_chunk()
        
        file_buffer.seek(0)
        return file_buffer.read()
    except Exception as e:
        print(f"파일 다운로드 실패: {str(e)}")
        return b''


def load_csv_with_encoding(content: bytes) -> pd.DataFrame:
    """CSV 파일을 DataFrame으로 로드 (인코딩 자동 감지)"""
    for encoding in ['utf-8', 'cp949', 'euc-kr']:
        try:
            return pd.read_csv(io.BytesIO(content), encoding=encoding)
        except (UnicodeDecodeError, Exception):
            continue
    return pd.DataFrame()


def get_latest_inventory() -> pd.DataFrame:
    """최신 Google Sheets 또는 CSV 파일에서 재고 데이터 로드 (10분 캐싱)"""
    cache_key = "inventory"
    
    if cache_key in inventory_cache:
        return inventory_cache[cache_key]
    
    drive_service = get_drive_service()
    if not drive_service:
        return pd.DataFrame()
    
    # Google Sheets 먼저 검색
    sheets_files = search_files_in_folder(
        mime_type='application/vnd.google-apps.spreadsheet',
        order_by='createdTime desc'
    )
    
    # CSV 파일도 검색
    csv_files = search_files_in_folder(
        mime_type='text/csv',
        order_by='createdTime desc'
    )
    
    # 모든 파일을 합쳐서 가장 최신 파일 선택
    all_files = sheets_files + csv_files
    if not all_files:
        return pd.DataFrame()
    
    # createdTime 기준 정렬
    all_files.sort(key=lambda x: x.get('createdTime', ''), reverse=True)
    latest_file = all_files[0]
    
    print(f"[재고] 최신 파일 로드: {latest_file['name']} (타입: {latest_file['mimeType']})")
    
    try:
        # Google Sheets인 경우 CSV로 내보내기
        if latest_file['mimeType'] == 'application/vnd.google-apps.spreadsheet':
            request = drive_service.files().export_media(
                fileId=latest_file['id'],
                mimeType='text/csv'
            )
            file_buffer = io.BytesIO()
            from googleapiclient.http import MediaIoBaseDownload
            downloader = MediaIoBaseDownload(file_buffer, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            file_buffer.seek(0)
            content = file_buffer.read()
        else:
            # CSV 파일인 경우 직접 다운로드
            content = download_file_content(latest_file['id'])
        
        if not content:
            return pd.DataFrame()
        
        df = load_csv_with_encoding(content)
        inventory_cache[cache_key] = df
        
        return df
    except Exception as e:
        print(f"[재고] 파일 로드 실패: {str(e)}")
        return pd.DataFrame()


def get_pdf_files_for_gemini() -> list:
    """모든 PDF 파일을 Gemini API에 업로드"""
    cache_key = "pdf_files"
    
    if cache_key in pdf_cache:
        return pdf_cache[cache_key]
    
    pdf_files = search_files_in_folder(mime_type='application/pdf')
    
    if not pdf_files:
        return []
    
    uploaded_files = []
    
    for pdf_file in pdf_files:
        print(f"[PDF] 업로드 중: {pdf_file['name']}")
        
        content = download_file_content(pdf_file['id'])
        if not content:
            continue
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            uploaded = genai.upload_file(tmp_path, mime_type='application/pdf')
            uploaded_files.append(uploaded)
            print(f"[PDF] 업로드 완료: {pdf_file['name']}")
        except Exception as e:
            print(f"[PDF] 업로드 실패: {str(e)}")
        finally:
            os.unlink(tmp_path)
    
    pdf_cache[cache_key] = uploaded_files
    return uploaded_files


def search_inventory(product_name: str) -> str:
    """재고 검색 후 결과 문자열 반환"""
    df = get_latest_inventory()
    
    if df.empty:
        return "재고 데이터를 불러올 수 없습니다."
    
    # 제품명/수량 컬럼 찾기
    product_col = None
    quantity_col = None
    
    for col in df.columns:
        col_lower = col.lower().strip()
        # 제품 컬럼 찾기 (Model Code, Model Number 등)
        if col_lower in ['model code', 'model number', 'model', 'code', 'product', 'item']:
            product_col = col
        if '제품' in col or '품명' in col or '상품' in col or '품번' in col:
            product_col = col
        # 수량 컬럼 찾기
        if col_lower == 'quantity' or col_lower == 'qty' or col_lower == 'stock':
            quantity_col = col
        if '수량' in col or '재고' in col:
            quantity_col = col
    
    if product_col is None:
        # Model Code가 없으면 첫 번째 컬럼이 아닌 두 번째 컬럼 시도 (Category 다음)
        if len(df.columns) > 1:
            product_col = df.columns[1]
        else:
            product_col = df.columns[0]
    if quantity_col is None and len(df.columns) > 3:
        quantity_col = df.columns[3]  # Quantity는 4번째 컬럼
    
    # 검색
    mask = df[product_col].astype(str).str.contains(product_name, case=False, na=False)
    matches = df[mask]
    
    if matches.empty:
        return f"'{product_name}' 제품을 찾을 수 없습니다."
    
    # 결과 포맷팅
    results = []
    for _, row in matches.head(5).iterrows():
        name = row[product_col]
        qty = row[quantity_col] if quantity_col else "정보 없음"
        results.append(f"• {name}: {qty}개")
    
    return "📦 재고 조회 결과\n\n" + "\n".join(results)


def chat_with_pdf(message: str) -> str:
    """PDF 기반 Gemini 챗봇 응답"""
    try:
        pdf_files = get_pdf_files_for_gemini()
        
        if not pdf_files:
            return "PDF 문서를 찾을 수 없습니다."
        
        model = genai.GenerativeModel(
            model_name='gemini-2.0-flash',
            generation_config={
                'temperature': 0.1,
                'max_output_tokens': 1024,
            }
        )
        
        system_prompt = """너는 기술 지원 AI야. 
반드시 첨부된 PDF 문서들의 내용에 기반해서만 답변해. 
문서에 없는 내용은 "해당 정보는 제공된 문서에서 찾을 수 없습니다."라고 답변해.
답변은 친절하고 명확하게, 카카오톡 메시지에 적합하게 간결하게 작성해."""
        
        contents = pdf_files + [f"{system_prompt}\n\n사용자 질문: {message}"]
        response = model.generate_content(contents)
        
        return response.text
        
    except Exception as e:
        print(f"챗봇 응답 생성 실패: {str(e)}")
        return "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다."


def make_kakao_response(text: str) -> Dict[str, Any]:
    """카카오 응답 포맷으로 변환"""
    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": text
                    }
                }
            ]
        }
    }


# ============ 카카오 스킬 엔드포인트 ============

@app.get("/")
async def root():
    """루트 경로"""
    return {"status": "ok", "message": "카카오톡 챗봇 서버가 실행 중입니다."}


@app.get("/health")
async def health_check():
    """헬스 체크 (카카오 오픈빌더 스킬 서버 확인용)"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/skill/inventory")
async def skill_inventory(request: KakaoRequest):
    """
    재고 확인 스킬
    
    카카오 오픈빌더에서 '재고' 인텐트로 연결
    params에 'product_name' 엔티티 필요
    """
    try:
        # 파라미터에서 제품명 추출
        product_name = request.action.params.get("product_name", "")
        
        # 파라미터가 없으면 발화에서 추출 시도
        if not product_name:
            utterance = request.userRequest.utterance
            # "재고", "수량" 등의 키워드 제거
            for keyword in ["재고", "수량", "몇개", "확인", "조회"]:
                utterance = utterance.replace(keyword, "")
            product_name = utterance.strip()
        
        if not product_name:
            return make_kakao_response("어떤 제품의 재고를 확인할까요?\n예: 'A제품 재고 확인'")
        
        result = search_inventory(product_name)
        return make_kakao_response(result)
        
    except Exception as e:
        print(f"재고 확인 오류: {str(e)}")
        return make_kakao_response("재고 확인 중 오류가 발생했습니다.")


@app.post("/skill/chat")
async def skill_chat(request: KakaoRequest):
    """
    PDF 기반 챗봇 스킬
    
    카카오 오픈빌더에서 '질문' 인텐트로 연결
    """
    try:
        message = request.userRequest.utterance
        
        if not message:
            return make_kakao_response("무엇이 궁금하신가요?")
        
        result = chat_with_pdf(message)
        return make_kakao_response(result)
        
    except Exception as e:
        print(f"챗봇 오류: {str(e)}")
        return make_kakao_response("응답 생성 중 오류가 발생했습니다.")


@app.post("/skill/fallback")
async def skill_fallback(request: KakaoRequest):
    """
    폴백 스킬 (기본 응답)
    
    매칭되는 인텐트가 없을 때 사용
    """
    message = request.userRequest.utterance
    
    # 재고 관련 키워드 확인
    inventory_keywords = ["재고", "수량", "몇개", "있어", "남아", "확인", "조회", "알려줘", "알려줘"]
    if any(kw in message for kw in inventory_keywords):
        # 키워드 제거 후 검색어 추출
        search_query = message
        for kw in inventory_keywords:
            search_query = search_query.replace(kw, "")
        search_query = search_query.strip()
        
        if search_query:
            result = search_inventory(search_query)
        else:
            result = "어떤 제품의 재고를 확인할까요?\n예: 'PAG-40-NE 재고'"
        return make_kakao_response(result)
    
    # 그 외는 PDF 기반 답변
    result = chat_with_pdf(message)
    return make_kakao_response(result)


# ============ 디버깅용 엔드포인트 ============

@app.get("/files")
async def list_files():
    """타겟 폴더 내의 모든 파일 목록 조회"""
    drive_service = get_drive_service()
    if not drive_service:
        return {"error": "Google Drive 연결 실패"}
    
    query = f"'{FOLDER_ID}' in parents and trashed=false"
    
    try:
        results = drive_service.files().list(
            q=query,
            fields='files(id, name, mimeType, createdTime)',
            orderBy='createdTime desc'
        ).execute()
        
        return {
            "folder_id": FOLDER_ID,
            "files": results.get('files', [])
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/cache/clear")
async def clear_cache():
    """캐시 초기화"""
    inventory_cache.clear()
    pdf_cache.clear()
    return {"message": "캐시가 초기화되었습니다."}


# 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

