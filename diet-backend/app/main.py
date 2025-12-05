from typing import List, Literal, Optional
import openai
from fastapi.middleware.cors import CORSMiddleware
import json
from fastapi import FastAPI, HTTPException,Query
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from sqlmodel import SQLModel, Field, create_engine, Session, select
import datetime
import os 
from dotenv import load_dotenv

app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")  

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

DATABASE_URL = "sqlite:///diet.db"
engine = create_engine(DATABASE_URL)

# MODELLER
class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    username: str
    hashed_password: str

class Meal(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str
    date: datetime.date
    meal_type: str  # "Kahvaltı", "Öğle", "Akşam"
    status: str     # "yendi", "atlandı", "alternatif"
    selected_meal: str
    calories: Optional[float] = 0
    protein: Optional[float] = 0
    carbs: Optional[float] = 0
    fat: Optional[float] = 0

@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)

# Pydantic Modeller
class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str

class LoginData(BaseModel):
    email: EmailStr
    password: str

class PersonalInfo(BaseModel):
    weight: float
    height: float
    age: int
    gender: str

class DietPreferences(BaseModel):
    diet_type: str
    exercise_level: str
    budget: str
    diet_preference: str

class UserInfo(BaseModel):
    personal_info: PersonalInfo
    diet_preferences: DietPreferences
    diet_goal: str
    disliked_foods: List[str]
    allergies: List[str]

class MealRecord(BaseModel):
    user_id: str
    date: str  # "2025-05-15"
    meal_type: Literal["Kahvaltı", "Ara Öğün", "Öğle Yemeği", "Akşam Yemeği"]
    status: Literal["yendi", "atlandı", "alternatif"]
    selected_meal: str
    calories: Optional[float] = 0
    protein: Optional[float] = 0
    carbs: Optional[float] = 0
    fat: Optional[float] = 0

# API ENDPOINTLERİ
class MacroData(BaseModel):
    protein: float
    carbs: float
    fat: float
    calories: Optional[float] = None 
class FeedbackData(BaseModel):
    user_id: str
    current_weight: float
    previous_weight: float
    hunger_level: int  # 1-5 arası
    fatigue_level: Optional[int] = None  # 1-5 arası
    satiety_level: Optional[int] = None  # 1-5 arası
    digestion_issues: Optional[bool] = False
    exercise: Optional[bool] = True
    feedback: str
    macro: MacroData

@app.post("/register")
async def register_user(user: UserCreate):
    with Session(engine) as session:
        if session.exec(select(User).where(User.email == user.email)).first():
            raise HTTPException(status_code=400, detail="E-posta zaten kayıtlı")
        hashed_password = pwd_context.hash(user.password)
        new_user = User(email=user.email, username=user.username, hashed_password=hashed_password)
        session.add(new_user)
        session.commit()
        session.refresh(new_user)  # user_id almak için ekle
        return {"message": "Kullanıcı başarıyla kaydedildi", "user_id": new_user.id}
@app.post("/login")
async def login_user(data: LoginData):
    with Session(engine) as session:
        user = session.exec(select(User).where(User.email == data.email)).first()
        if not user or not pwd_context.verify(data.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="E-posta veya şifre hatalı")
        return {"message": "Giriş başarılı", "username": user.username}

@app.post("/save_meal")
async def save_meal(record: MealRecord):
    try:
        meal = Meal(
            user_id=record.user_id,
            date=datetime.date.fromisoformat(record.date),
            meal_type=record.meal_type,
            status=record.status,
            selected_meal=record.selected_meal,
            calories=record.calories,
            protein=record.protein,
            carbs=record.carbs,
            fat=record.fat
        )
        with Session(engine) as session:
            session.add(meal)
            session.commit()
        return {"message": "Öğün başarıyla kaydedildi"}
    except Exception as e:
        return {"error": str(e)}



@app.get("/get_meals/{user_id}")
async def get_meals_by_user(user_id: str):
    try:
        with Session(engine) as session:
            meals = session.exec(select(Meal).where(Meal.user_id == user_id)).all()
        return {"meals": [meal.dict() for meal in meals]}
    except Exception as e:
        return {"error": str(e)}

@app.get("/get_totals/{user_id}")
async def get_total_macros(user_id: str):
    try:
        with Session(engine) as session:
            meals = session.exec(
                select(Meal).where(Meal.user_id == user_id, Meal.status == "yendi")
            ).all()
        total_calories = sum(m.calories or 0 for m in meals)
        total_protein = sum(m.protein or 0 for m in meals)
        total_carbs = sum(m.carbs or 0 for m in meals)
        total_fat = sum(m.fat or 0 for m in meals)
        return {
            "total_calories": total_calories,
            "total_protein": total_protein,
            "total_carbs": total_carbs,
            "total_fat": total_fat
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/generate_diet")
async def generate_diet(user: UserInfo):
    pi = user.personal_info
    dp = user.diet_preferences
    prompt = (
        f"Kullanıcı bilgileri:\n"
        f"- Kilo: {pi.weight} kg\n"
        f"- Boy: {pi.height} cm\n"
        f"- Yaş: {pi.age}\n"
        f"- Cinsiyet: {pi.gender}\n"
        f"- Diyet amacı: {user.diet_goal}\n"
        f"- Diyet türü: {dp.diet_type}\n"
        f"- Egzersiz seviyesi: {dp.exercise_level}\n"
        f"- Bütçe: {dp.budget}\n"
        f"- Diyet tercihi: {dp.diet_preference}\n"
        f"- Sevmediği yiyecekler: {', '.join(user.disliked_foods) if user.disliked_foods else 'Yok'}\n"
        f"- Alerjiler: {', '.join(user.allergies) if user.allergies else 'Yok'}\n\n"
        f"Lütfen aşağıdaki kurallara göre **haftalık** (Pazartesi, Salı, Çarşamba, Perşembe, Cuma, Cumartesi, Pazar) bir diyet planı oluştur:\n"
        f"- Her gün için gün adını kullan (ör: Pazartesi, Salı, ...).\n"
        f"- Her günün altında Kahvaltı, Ara Öğün, Öğle Yemeği, Akşam Yemeği gibi öğünler olsun.\n"
        f"- Her öğün için önerilen yemekleri, makro besin değerlerini (protein, karbonhidrat, yağ) ve kalori bilgisini belirt.\n"
        f"- Eğer diyet tercihi vegan veya vejetaryen ise kesinlikle hayvansal ürün önermeden plan oluştur.\n"
        f"- Cevabı yalnızca geçerli bir JSON formatında ver.\n"
        f"- Her öğün için mutlaka şu formatta bilgi ver:\n"
        f"  \"Yemek adı - Xg protein, Yg karbonhidrat, Zg yağ, N kalori\"\n"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
        )
        ai_response = response.choices[0].message["content"].strip()
        print("AI yanıtı:", ai_response)  # <-- Bunu ekle

        # Kod bloğu işaretlerini ve baştaki/sondaki açıklamaları temizle
        if ai_response.startswith("```json"):
            ai_response = ai_response[7:]
        if ai_response.startswith("```"):
            ai_response = ai_response[3:]
        if ai_response.endswith("```"):
            ai_response = ai_response[:-3]
        ai_response = ai_response.strip()

        # Eğer hala JSONDecodeError alırsan, aşağıdaki gibi sadece JSON kısmını almaya çalış:
        if not ai_response.startswith("{"):
            first_brace = ai_response.find("{")
            last_brace = ai_response.rfind("}")
            if first_brace != -1 and last_brace != -1:
                ai_response = ai_response[first_brace:last_brace+1]

        try:
            diet_dict = json.loads(ai_response.replace("'", '"'))
        except json.JSONDecodeError as e:
            return {"error": f"Yapay zeka JSON formatında cevap veremedi: {str(e)}", "raw_response": ai_response}

        return {"diet": diet_dict}
    except Exception as e:
        return {"error": str(e)}

@app.get("/summary/{user_id}")
async def get_summary(user_id: str):
    try:
        with Session(engine) as session:
            statement = select(Meal).where(
                Meal.user_id == user_id,
                Meal.status == "yendi"
            )
            meals = session.exec(statement).all()

            if not meals:
                return {
                    "user_id": user_id,
                    "totals": {
                        "calories": 0,
                        "protein": 0,
                        "carbs": 0,
                        "fat": 0,
                    }
                }

            total_calories = sum(meal.calories or 0 for meal in meals)
            total_protein = sum(meal.protein or 0 for meal in meals)
            total_carbs = sum(meal.carbs or 0 for meal in meals)
            total_fat = sum(meal.fat or 0 for meal in meals)

        return {
            "user_id": user_id,
            "totals": {
                "calories": total_calories,
                "protein": total_protein,
                "carbs": total_carbs,
                "fat": total_fat,
            }
        }

    except Exception as e:
        return {"error": str(e)}




def generate_prompt(data: FeedbackData) -> str:
    macro_warning = ""
    extra_snack = ""
    ekstra_oneri = ""

    toplam_kalori = data.macro.calories
    if not toplam_kalori:
        toplam_kalori = (
            data.macro.protein * 4 +
            data.macro.carbs * 4 +
            data.macro.fat * 9
        )

    if data.hunger_level > 3:
        extra_snack += "- Kullanıcı sık acıkıyor. Bir adet sağlıklı ara öğün ekle.\n"
    if data.current_weight >= data.previous_weight:
        if data.macro.fat > 70 or data.macro.carbs > 200:
            macro_warning += "- Kullanıcının kilo veremediği görülüyor. Karbonhidrat ve yağ oranını azalt, protein oranını artır.\n"
    if data.fatigue_level and data.fatigue_level > 3:
        ekstra_oneri += "- Kullanıcı kendini yorgun hissediyor. Karbonhidrat miktarını biraz artır.\n"
    if data.digestion_issues:
        ekstra_oneri += "- Kullanıcı sindirim problemleri yaşıyor. Lif oranını artır.\n"
    if data.exercise:
        ekstra_oneri += "- Kullanıcı egzersiz yapıyor. Protein miktarını artır.\n"

    prompt = f"""
Sen uzman bir diyetisyensin. Aşağıdaki kullanıcı verilerine göre kişiye özel yeni bir diyet önerisi oluştur:

- Önceki kilo: {data.previous_weight} kg
- Şu anki kilo: {data.current_weight} kg
- Açlık seviyesi (1-5): {data.hunger_level}
- Tatmin seviyesi (1-5): {data.satiety_level}
- Yorgunluk seviyesi (1-5): {data.fatigue_level}
- Sindirim problemi var mı?: {"Evet" if data.digestion_issues else "Hayır"}
- Egzersiz yapıyor mu?: {"Evet" if data.exercise else "Hayır"}
- Kullanıcı geri bildirimi: "{data.feedback}"
- Önceki makrolar: Protein: {data.macro.protein}g, Karbonhidrat: {data.macro.carbs}g, Yağ: {data.macro.fat}g
- Önceki toplam kalori: {toplam_kalori} kcal

{extra_snack}{macro_warning}{ekstra_oneri}
Yemek listesi örneğiyle birlikte yeni bir haftalık diyet önerisi oluştur. 
Toplam makro değerlerini ve alınan kaloriyi dikkate alarak öneri yap.
"""

    return prompt.strip()



@app.post("/improve_diet/")
async def improve_diet(data: FeedbackData):
    prompt = generate_prompt(data)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Sen uzman bir diyetisyensin."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=5000
        )
        new_diet = response["choices"][0]["message"]["content"]
        return {"suggested_diet": new_diet}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))