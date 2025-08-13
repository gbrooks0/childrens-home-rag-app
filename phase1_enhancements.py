# STEP 1: Save this entire file as "phase1_enhancements.py" in your project root

import re
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time

# =============================================================================
# ENTITY EXTRACTION - Extract Real Provider Names from Ofsted Reports
# =============================================================================

@dataclass
class ExtractedEntities:
    """Enhanced entity extraction for template population"""
    provider_names: List[str]
    inspection_dates: List[str]
    ratings: Dict[str, str]
    comparison_context: str
    confidence_score: float
    
    def has_comparison_data(self) -> bool:
        return len(self.provider_names) >= 2
    
    def get_provider_display_names(self) -> List[str]:
        """Get cleaned provider names for display"""
        cleaned = []
        for name in self.provider_names:
            # Clean up common variations
            clean_name = name.strip()
            if clean_name and len(clean_name) > 3:
                cleaned.append(clean_name)
        return cleaned[:2]  # Max 2 for comparison

class EntityExtractor:
    """Extract real provider names from Ofsted analysis to fix template placeholders"""
    
    def __init__(self):
        self.provider_patterns = [
            r'([A-Z][a-z]+ [A-Z][a-z]+ Homes? (?:Ltd|Limited|LLP))',
            r'([A-Z][a-z]+ [A-Z][a-z]+ Care (?:Ltd|Limited))',
            r'(Creating Lifestyles Homes Limited)',
            r'(Yorkshire Serenity Care Ltd)',
            r'(Restoring Lives Ltd)',
        ]
    
    def extract_from_ofsted_cache(self, rag_system):
        """Extract entities from cached Ofsted analysis - OPTIMIZED"""
        try:
            # Check if already extracted in this session
            if hasattr(self, '_cached_entities_session'):
                print("⚡ USING CACHED entities from this session")
                return self._cached_entities_session
            
            # Try multiple sources for Ofsted data
            ofsted_analysis = getattr(rag_system, '_last_ofsted_analysis', None)
            
            # FALLBACK: If no ofsted_analysis, try to extract from session state
            if not ofsted_analysis or not ofsted_analysis.get('has_ofsted'):
                print("🔍 FALLBACK: Extracting from session cached_file_content")
                import streamlit as st
                if hasattr(st, 'session_state') and hasattr(st.session_state, 'cached_file_content'):
                    cached_files = st.session_state.cached_file_content
                    if cached_files:
                        return self._extract_from_cached_files(cached_files)
                return self._create_empty_entities()

            print(f"🔍 EXTRACTING entities from {len(ofsted_analysis.get('ofsted_reports', []))} reports")
            
            provider_names = []
            ratings = {}
            provider_ratings = {}
            inspection_dates = []
            
            # Extract from cached ofsted_reports structure
            for report in ofsted_analysis.get('ofsted_reports', []):
                summary = report.get('summary')
                if summary:
                    if hasattr(summary, 'provider_name') and summary.provider_name:
                        if summary.provider_name != "Unknown Provider":
                            provider_name = summary.provider_name
                            provider_names.append(provider_name)
                    
                    if hasattr(summary, 'overall_rating') and summary.overall_rating and provider_name:
                        ratings[provider_name] = summary.overall_rating
                        provider_ratings[provider_name] = summary.overall_rating
                    
                    if hasattr(summary, 'inspection_date') and summary.inspection_date:
                        inspection_dates.append(str(summary.inspection_date))
            
            comparison_context = 'provider_comparison' if len(provider_names) >= 2 else 'single_provider'
            confidence = 0.9 if provider_names else 0.3
            
            print(f"🔍 ENTITIES EXTRACTED: {provider_names}")
            print(f"🎯 PROVIDER RATINGS: {provider_ratings}")
            
            entities = ExtractedEntities(
                provider_names=list(set(provider_names)),
                inspection_dates=inspection_dates,
                ratings=ratings,
                comparison_context=comparison_context,
                confidence_score=confidence
            )
            
            # Cache for this session
            self._cached_entities_session = entities
            self._current_provider_ratings = provider_ratings
            
            return entities
            
        except Exception as e:
            print(f"❌ Entity extraction failed: {e}")
            return self._create_empty_entities()
    
    def _extract_from_cached_files(self, cached_files):
        """Extract entities directly from cached file content"""
        provider_names = []
        ratings = {}
        
        for filename, content in cached_files.items():
            if 'ofsted' in filename.lower() or '.pdf' in filename.lower():
                # Extract provider name from content
                for pattern in self.provider_patterns:
                    match = re.search(pattern, content)
                    if match:
                        provider_name = match.group(1).strip()
                        provider_names.append(provider_name)
                        
                        # Simple rating extraction
                        if 'requires improvement' in content.lower():
                            ratings[provider_name] = 'Requires improvement'
                        elif 'good' in content.lower():
                            ratings[provider_name] = 'Good'
                        
                        print(f"🔍 EXTRACTED from cache: {provider_name}")
        
        return ExtractedEntities(
            provider_names=list(set(provider_names)),
            inspection_dates=[],
            ratings=ratings,
            comparison_context='provider_comparison' if len(provider_names) >= 2 else 'single_provider',
            confidence_score=0.8 if provider_names else 0.0
        )
    

    def _create_empty_entities(self) -> ExtractedEntities:
        """Create empty entities for fallback"""
        return ExtractedEntities(
            provider_names=[],
            inspection_dates=[],
            ratings={},
            comparison_context='unknown',
            confidence_score=0.0
        )

# =============================================================================
# TEMPLATE VALIDATOR - Prevent Placeholder Issues
# =============================================================================

class TemplateValidator:
    """Validate template data and prevent placeholder issues"""
    
    def validate_comparison_template(self, entities: ExtractedEntities) -> Dict[str, Any]:
        """Validate data for comparison templates"""
        
        provider_names = entities.get_provider_display_names()
        
        validation_result = {
            'is_valid': len(provider_names) >= 2,
            'provider_names': provider_names,
            'fallback_needed': False,
            'validation_message': ''
        }
        
        if not validation_result['is_valid']:
            validation_result['fallback_needed'] = True
            if len(provider_names) == 1:
                validation_result['validation_message'] = f'Only one provider found: {provider_names[0]}'
            else:
                validation_result['validation_message'] = 'No provider names found in source documents'
        else:
            validation_result['validation_message'] = f'Comparison ready: {" vs ".join(provider_names)}'
        
        print(f"🔍 TEMPLATE VALIDATION: {validation_result}")
        return validation_result

# =============================================================================
# ENHANCED QUERY PROCESSOR - Integration with Your RAG System
# =============================================================================

class OptimizedEnhancedQueryProcessor:
    """LIGHTWEIGHT processor that only runs when needed"""
    
    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.template_validator = TemplateValidator()
        self._provider_ratings_cache = {}  # Cache ratings
    
    def enhance_response_generation(self, original_result: Dict, rag_system, question: str) -> Dict:
        """OPTIMIZED enhancement - minimal processing"""
        
        try:
            answer = original_result.get('answer', '')
            
            # FAST EXIT: Only process comparison queries with placeholders
            if not self._needs_comparison_fix(answer, question):
                return original_result
            
            # Extract entities ONCE and cache
            entities = self._get_cached_entities_or_extract(rag_system)
            
            if entities.has_comparison_data():
                print(f"🎯 FIXING placeholders for: {entities.provider_names}")
                
                # Get cached ratings
                provider_ratings = self._get_provider_ratings_from_cache(entities)
                
                # Fix placeholders efficiently
                fixed_answer = self._fix_placeholders_fast(
                    answer, entities.provider_names, provider_ratings
                )
                
                original_result['answer'] = fixed_answer
                original_result['enhancement_data'] = {
                    'placeholders_fixed': True,
                    'providers': entities.provider_names,
                    'confidence': entities.confidence_score
                }
            
            return original_result
            
        except Exception as e:
            print(f"❌ Enhancement error: {e}")
            return original_result
    
    def _needs_comparison_fix(self, answer: str, question: str) -> bool:
        """FAST check for placeholder issues"""
        has_placeholders = '[HIGHER-RATED' in answer or '[LOWER-RATED' in answer
        is_comparison = 'compar' in question.lower()
        return has_placeholders and is_comparison
    
    def _fix_comparison_placeholders(self, answer: str, provider_names: List[str]) -> str:
        """Replace template placeholders with real provider names AND correct ratings"""
        
        if len(provider_names) < 2:
            return answer
        
        # Get the actual ratings from Ofsted analysis instead of guessing
        provider_ratings = self._get_provider_ratings_from_entities()
        
        # Determine correct provider-rating mapping
        higher_provider, lower_provider = self._determine_correct_rating_order(
            provider_names, provider_ratings, answer
        )
        
        print(f"🎯 RATING MAPPING: Higher={higher_provider}, Lower={lower_provider}")
        
        # Replace placeholders with correct mapping
        fixed_answer = answer.replace('[HIGHER-RATED PROVIDER NAME]', higher_provider)
        fixed_answer = fixed_answer.replace('[LOWER-RATED PROVIDER NAME]', lower_provider)
        fixed_answer = fixed_answer.replace('[HIGHER-RATED HOME]', higher_provider)
        fixed_answer = fixed_answer.replace('[LOWER-RATED HOME]', lower_provider)
        
        return fixed_answer
    
    def _get_provider_ratings_from_entities(self) -> Dict[str, str]:
        """Extract actual provider ratings from cached entities"""
        # This will be populated during entity extraction
        return getattr(self, '_current_provider_ratings', {})

    def _determine_correct_rating_order(self, provider_names: List[str], 
                                      provider_ratings: Dict[str, str], 
                                      answer: str) -> Tuple[str, str]:
        """Determine which provider is higher rated based on actual Ofsted ratings"""
        
        provider_a, provider_b = provider_names[0], provider_names[1]
        
        # Rating hierarchy (higher index = better rating)
        rating_order = ['Inadequate', 'Requires improvement', 'Requires Improvement', 'Good', 'Outstanding']
        
        # Get ratings for each provider
        rating_a = provider_ratings.get(provider_a, '')
        rating_b = provider_ratings.get(provider_b, '')
        
        print(f"🔍 ACTUAL RATINGS: {provider_a}={rating_a}, {provider_b}={rating_b}")
        
        # Compare ratings
        try:
            # Handle "Requires Improvement to be Good" special case
            if 'Requires Improvement' in rating_a:
                rating_a_clean = 'Requires improvement'
            else:
                rating_a_clean = rating_a
                
            if 'Requires Improvement' in rating_b:
                rating_b_clean = 'Requires improvement'
            else:
                rating_b_clean = rating_b
            
            # Find positions in rating hierarchy
            index_a = rating_order.index(rating_a_clean) if rating_a_clean in rating_order else 1
            index_b = rating_order.index(rating_b_clean) if rating_b_clean in rating_order else 1
            
            # Higher index = better rating
            if index_a > index_b:
                return provider_a, provider_b  # A is higher
            elif index_b > index_a:
                return provider_b, provider_a  # B is higher
            else:
                # Same rating - use text analysis as fallback
                return self._fallback_rating_determination(provider_a, provider_b, answer)
                
        except (ValueError, IndexError):
            # Fallback to text analysis if rating comparison fails
            return self._fallback_rating_determination(provider_a, provider_b, answer)

    def _get_cached_entities_or_extract(self, rag_system) -> ExtractedEntities:
        """Get entities from cache or extract once"""
        
        # Check if already cached for this session
        if hasattr(self, '_cached_entities'):
            print("⚡ USING cached entities")
            return self._cached_entities
        
        # Extract once and cache
        print("🔍 EXTRACTING entities (first time)")
        entities = self.entity_extractor.extract_from_ofsted_cache(rag_system)
        self._cached_entities = entities
        
        return entities
    
    def _get_provider_ratings_from_cache(self, entities: ExtractedEntities) -> Dict[str, str]:
        """Get provider ratings efficiently"""
        return entities.ratings
    
    def _fix_placeholders_fast(self, answer: str, provider_names: List[str], 
                             provider_ratings: Dict[str, str]) -> str:
        
        try:
            # Your existing logic...
            if len(provider_names) < 2:
                return answer
            
            provider_a, provider_b = provider_names[0], provider_names[1]
            rating_a = provider_ratings.get(provider_a, 'Unknown')
            rating_b = provider_ratings.get(provider_b, 'Unknown')
            
            print(f"⚡ RATING LOGIC: {provider_a}={rating_a}, {provider_b}={rating_b}")
            
            # Set defaults and logic...
            higher_provider, lower_provider = provider_a, provider_b
            
            if rating_a == 'Good' and rating_b == 'Requires improvement':
                higher_provider, lower_provider = provider_a, provider_b
            elif rating_b == 'Good' and rating_a == 'Requires improvement':
                higher_provider, lower_provider = provider_b, provider_a
            
            print(f"✅ FINAL MAPPING: Higher={higher_provider}, Lower={lower_provider}")
            
            # ADD ERROR HANDLING HERE:
            print(f"🔍 BEFORE replacement: Answer length = {len(answer)}")
            
            # Try replacement with safety check
            if '[HIGHER-RATED PROVIDER NAME]' in answer:
                answer = answer.replace('[HIGHER-RATED PROVIDER NAME]', higher_provider)
            if '[LOWER-RATED PROVIDER NAME]' in answer:
                answer = answer.replace('[LOWER-RATED PROVIDER NAME]', lower_provider)
            if '[HIGHER-RATED HOME]' in answer:
                answer = answer.replace('[HIGHER-RATED HOME]', higher_provider)
            if '[LOWER-RATED HOME]' in answer:
                answer = answer.replace('[LOWER-RATED HOME]', lower_provider)
            
            print(f"🔍 AFTER replacement: Answer length = {len(answer)}")
            return answer
            
        except Exception as e:
            print(f"❌ Placeholder replacement failed: {e}")
            print(f"❌ Returning original answer unchanged")
            return answer  # Return original if replacement fails

    def _fallback_rating_determination(self, provider_a: str, provider_b: str, 
                                     answer: str) -> Tuple[str, str]:
        """Fallback method using text analysis"""
        
        # Look for explicit statements in the text
        if f"{provider_a} (Good)" in answer and f"{provider_b} (Requires Improvement" in answer:
            return provider_a, provider_b
        elif f"{provider_b} (Good)" in answer and f"{provider_a} (Requires Improvement" in answer:
            return provider_b, provider_a
        else:
            # Default to first provider as higher (conservative approach)
            print(f"⚠️ FALLBACK: Using {provider_a} as higher rated (default)")
            return provider_a, provider_b

    def _add_data_limitation_disclaimer(self, answer: str, message: str) -> str:
        """Add disclaimer when we can't fix placeholders"""
        
        disclaimer = f"\n\n**⚠️ Data Limitation Notice:**\n{message}\n\nPlease refer to the specific inspection reports for accurate provider names and details."
        
        return answer + disclaimer

# =============================================================================
# MAIN INTEGRATION FUNCTION
# =============================================================================

def enhance_your_rag_system(rag_system):
    """
    OPTIMIZED INTEGRATION FUNCTION - Prevents multiple integrations
    """
    
    # CRITICAL: Check if already enhanced to prevent multiple integrations
    if hasattr(rag_system, '_phase1_enhanced'):
        print("⚡ Phase 1 already integrated - skipping duplicate integration")
        return rag_system
    
    print("🚀 INTEGRATING Phase 1 enhancements...")
    
    # Add the enhanced processor ONCE - FIXED CLASS NAME
    if not hasattr(rag_system, 'enhanced_processor'):
        rag_system.enhanced_processor = OptimizedEnhancedQueryProcessor()  # CHANGED: Use correct class name
        print("✅ Optimized processor added")
    
    # Store original query method for fallback ONCE
    if not hasattr(rag_system, '_original_query'):
        rag_system._original_query = rag_system.query
        print("✅ Original query method backed up")
    
    # Create LIGHTWEIGHT enhanced query method
    def enhanced_query(question, k=5, response_style="standard", performance_mode="balanced",
                      is_file_analysis=False, uploaded_files=None, uploaded_images=None):
        """OPTIMIZED enhanced version of your existing query method"""
        
        # Get your existing results first (unchanged)
        original_result = rag_system._original_query(
            question=question,
            k=k,
            response_style=response_style,
            performance_mode=performance_mode,
            is_file_analysis=is_file_analysis,
            uploaded_files=uploaded_files,
            uploaded_images=uploaded_images
        )
        
        # LIGHTWEIGHT enhancement - only for comparison queries
        try:
            answer = original_result.get('answer', '')
            
            # FAST CHECK: Only enhance if placeholders detected
            if ('[HIGHER-RATED' in answer or '[LOWER-RATED' in answer):
                print("🔧 APPLYING enhancement (placeholders detected)")
                enhanced_result = rag_system.enhanced_processor.enhance_response_generation(
                    original_result, rag_system, question
                )
                return enhanced_result
            else:
                print("⚡ SKIPPING enhancement (no placeholders)")
                return original_result
            
        except Exception as e:
            print(f"⚠️ Enhancement failed, using original: {e}")
            return original_result
    
    # Replace the query method
    rag_system.query = enhanced_query
    
    # Mark as enhanced to prevent re-integration
    rag_system._phase1_enhanced = True
    print("✅ OPTIMIZED Phase 1 integration complete")
    
    return rag_system

# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

def get_enhancement_performance_stats(rag_system):
    """Get performance statistics for Phase 1 enhancements"""
    
    stats = {
        'integration_active': hasattr(rag_system, '_phase1_enhanced'),
        'processor_type': type(rag_system.enhanced_processor).__name__ if hasattr(rag_system, 'enhanced_processor') else 'None',
        'entities_cached': hasattr(rag_system.enhanced_processor, '_cached_entities') if hasattr(rag_system, 'enhanced_processor') else False,
        'multiple_integrations_prevented': hasattr(rag_system, '_phase1_enhanced')
    }
    
    return stats





# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def test_enhancement_integration(rag_system):
    """Test if Phase 1 enhancements are working"""
    
    test_results = {
        'integration_active': hasattr(rag_system, 'enhanced_processor'),
        'original_backed_up': hasattr(rag_system, '_original_query'),
        'entity_extractor_working': False,
        'template_validator_working': False
    }
    
    try:
        # Test entity extraction
        entities = rag_system.enhanced_processor.entity_extractor.extract_from_ofsted_cache(rag_system)
        test_results['entity_extractor_working'] = entities.confidence_score > 0
        test_results['providers_found'] = entities.provider_names
        
        # Test template validation
        validation = rag_system.enhanced_processor.template_validator.validate_comparison_template(entities)
        test_results['template_validator_working'] = 'is_valid' in validation
        test_results['validation_result'] = validation
        
    except Exception as e:
        test_results['error'] = str(e)
    
    return test_results
