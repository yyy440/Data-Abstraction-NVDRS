from pydantic import BaseModel, Field

location_map = {1: "House, apartment",
        2: "Motor vehicle (excluding school bus and public transportation)",
        3: "Natural area (e.g., field, river, beaches, woods)",
        4: "Park, playground, public use area",
        5: "Street/road, sidewalk, alley",
        6: "Other"
    }
weapon_map = {1: "Blunt instrument",
                2: "Drowning",
                3: "Fall",
                4: "Fire or burns",
                5: "Firearm",
                6: "Hanging, strangulation, suffocation",
                7: "Motor vehicle including buses, motorcycles",
                8: "Other transport vehicle, eg, trains, planes, boats",
                9: "Poisoning",
                10: "Sharp instrument",
                11: "Other (e.g. taser, electrocution, nail gun)",
                12: "Unknown"
            }



class FirstSchema(BaseModel):
    DepressedMood: int
    MentalIllnessTreatmentCurrnt: int
    HistoryMentalIllnessTreatmnt: int
    SuicideAttemptHistory: int

class SecondSchema(BaseModel):
    SuicideThoughtHistory: int
    SubstanceAbuseProblem: int
    MentalHealthProblem: int
    DiagnosisAnxiety: int


class ThirdSchema(BaseModel):
    DiagnosisDepressionDysthymia: int
    DiagnosisBipolar: int
    DiagnosisAdhd: int
    IntimatePartnerProblem: int
    
class FourthSchema(BaseModel):
    FamilyRelationship: int
    Argument: int
    SchoolProblem: int
    RecentCriminalLegalProblem: int


class FifthSchema(BaseModel):
    SuicideNote: int
    SuicideIntentDisclosed: int
    DisclosedToIntimatePartner: int
    DisclosedToOtherFamilyMember: int
        
class SixthSchema(BaseModel):
    
    DisclosedToFriend: int
    InjuryLocationType: int
    WeaponType: int

class DifficultSchema(BaseModel):

    FamilyRelationship: int
    RecentCriminalLegalProblem: int
    DisclosedToIntimatePartner: int 
    DisclosedToOtherFamilyMember: int
    DisclosedToFriend: int

class AllSchema(BaseModel):

    DepressedMood: int
    MentalIllnessTreatmentCurrnt: int
    HistoryMentalIllnessTreatmnt: int
    SuicideAttemptHistory: int
    SuicideThoughtHistory: int
    SubstanceAbuseProblem: int
    MentalHealthProblem: int
    DiagnosisAnxiety: int
    DiagnosisDepressionDysthymia: int
    DiagnosisBipolar: int
    DiagnosisAdhd: int
    IntimatePartnerProblem: int
    FamilyRelationship: int
    Argument: int
    SchoolProblem: int
    RecentCriminalLegalProblem: int
    SuicideNote: int
    SuicideIntentDisclosed: int
    DisclosedToIntimatePartner: int
    DisclosedToOtherFamilyMember: int
    DisclosedToFriend: int
    InjuryLocationType: int
    WeaponType1: int

