#!/usr/bin/env python3
"""
Setup and Test Script for Smart Index Selection RAG System

This script helps you set up and test the dual index RAG system with smart routing.
Run this after creating your dual indexes to verify everything works correctly.
"""

import os
import sys
import time
from pathlib import Path
import json

def check_environment():
    """Check if required environment variables are set."""
    print("🔍 Checking environment variables...")
    
    required_vars = ["OPENAI_API_KEY", "GOOGLE_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if var not in os.environ:
            missing_vars.append(var)
        else:
            print(f"  ✅ {var}: Set")
    
    if missing_vars:
        print(f"\n❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these variables before running the system.")
        return False
    
    print("✅ All environment variables are set!")
    return True

def check_indexes():
    """Check if the dual indexes exist."""
    print("\n🗂️  Checking for dual indexes...")
    
    index_paths = {
        "OpenAI": "indexes/openai_index",
        "Google": "indexes/google_index"
    }
    
    existing_indexes = []
    missing_indexes = []
    
    for name, path in index_paths.items():
        if Path(path).exists():
            # Check if it contains FAISS files
            faiss_files = list(Path(path).glob("*.faiss")) + list(Path(path).glob("*.pkl"))
            if faiss_files:
                print(f"  ✅ {name} index: Found at {path}")
                existing_indexes.append(name)
            else:
                print(f"  ⚠️  {name} index: Directory exists but no FAISS files found")
                missing_indexes.append(name)
        else:
            print(f"  ❌ {name} index: Not found at {path}")
            missing_indexes.append(name)
    
    if not existing_indexes:
        print("\n❌ No indexes found! Please run the enhanced ingestion script first.")
        return False, []
    
    if missing_indexes:
        print(f"\n⚠️  Some indexes are missing: {', '.join(missing_indexes)}")
        print("The system will work with available indexes but may have limited fallback options.")
    
    return True, existing_indexes

def test_smart_router():
    """Test the smart router functionality."""
    print("\n🧠 Testing Smart Router...")
    
    try:
        from smart_query_router import SmartRouter
        
        router = SmartRouter()
        
        # Test query analysis
        test_queries = [
            "What are the safeguarding policies for children's homes?",
            "How do I implement error handling in Python?",
            "What are the legal requirements for fostering agencies?"
        ]
        
        print("  📝 Testing query analysis...")
        for query in test_queries[:2]:  # Test first 2 queries
            analysis = router.query_analyzer.analyze_query(query)
            print(f"    Query: '{query[:40]}...'")
            print(f"    Pattern: {analysis['best_pattern']}")
            print(f"    Preferred Provider: {analysis['preferred_provider']}")
            print(f"    Confidence: {analysis['pattern_confidence']:.2f}")
        
        # Test routing
        print("  🔄 Testing document retrieval...")
        result = router.route_query(test_queries[0], k=3)
        
        if result["success"]:
            print(f"    ✅ Retrieved {result['total_results']} documents")
            print(f"    Provider used: {result['provider']}")
            print(f"    Response time: {result['response_time']:.3f}s")
        else:
            print(f"    ❌ Retrieval failed: {result['error']}")
            return False
        
        print("✅ Smart Router is working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Cannot import smart router: {e}")
        return False
    except Exception as e:
        print(f"❌ Smart router test failed: {e}")
        return False

def test_rag_system():
    """Test the enhanced RAG system."""
    print("\n🤖 Testing Enhanced RAG System...")
    
    try:
        from rag_system import EnhancedRAGSystem
        
        # Test with OpenAI LLM
        print("  🔧 Initializing RAG system...")
        rag = EnhancedRAGSystem(llm_provider="openai")
        
        # Test query
        test_question = "What are the key safeguarding policies for children's homes?"
        print(f"  🔍 Testing query: '{test_question}'")
        
        start_time = time.time()
        result = rag.query(test_question, k=3)
        end_time = time.time()
        
        if result.get("success", True):
            print(f"    ✅ Query successful!")
            print(f"    Response time: {end_time - start_time:.2f}s")
            print(f"    Confidence: {result['confidence_score']:.2f}")
            print(f"    Sources: {result['total_sources']}")
            print(f"    Embedding provider: {result['routing_info']['embedding_provider']}")
            print(f"    Answer length: {len(result['answer'])} characters")
            
            # Show first part of answer
            answer_preview = result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer']
            print(f"    Answer preview: {answer_preview}")
            
        else:
            print(f"    ❌ Query failed: {result.get('error')}")
            return False
        
        print("✅ Enhanced RAG System is working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Cannot import RAG system: {e}")
        return False
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        return False

def test_embedding_comparison():
    """Test comparison between different embedding providers."""
    print("\n🔄 Testing Embedding Provider Comparison...")
    
    try:
        from rag_system import EnhancedRAGSystem
        
        rag = EnhancedRAGSystem()
        test_question = "What are the inspection requirements for children's homes?"
        
        print(f"  🔍 Comparing embeddings for: '{test_question}'")
        comparison = rag.compare_embeddings(test_question, k=3)
        
        print("  📊 Comparison Results:")
        for provider, results in comparison["provider_comparison"].items():
            print(f"    {provider.upper()}:")
            print(f"      Confidence: {results['confidence']:.2f}")
            print(f"      Response Time: {results['response_time']:.2f}s")
            print(f"      Sources Found: {results['source_count']}")
            if results['used_fallback']:
                print(f"      ⚠️  Used fallback")
        
        print("✅ Embedding comparison completed!")
        return True
        
    except Exception as e:
        print(f"❌ Embedding comparison test failed: {e}")
        return False

def show_system_status():
    """Show current system status and metrics."""
    print("\n📊 System Status and Configuration...")
    
    try:
        from rag_system import EnhancedRAGSystem
        
        rag = EnhancedRAGSystem()
        status = rag.get_system_status()
        
        print("  🔧 System Configuration:")
        print(f"    LLM Provider: {status['system_info']['llm_provider']}")
        print(f"    Available Embedding Providers: {', '.join(status['system_info']['embedding_providers_available'])}")
        
        print("  📈 Performance Metrics:")
        print(f"    Total Queries: {status['performance_metrics']['total_queries']}")
        if status['performance_metrics']['total_queries'] > 0:
            success_rate = (status['performance_metrics']['successful_queries'] / 
                          status['performance_metrics']['total_queries']) * 100
            print(f"    Success Rate: {success_rate:.1f}%")
            print(f"    Average Response Time: {status['performance_metrics']['average_response_time']:.2f}s")
        
        print("  🗂️  Router Status:")
        router_perf = status['router_performance']
        for provider, metrics in router_perf.get('providers', {}).items():
            print(f"    {provider.upper()}:")
            print(f"      Performance Score: {metrics['score']:.2f}")
            print(f"      Success Rate: {metrics['success_rate']:.2f}")
            print(f"      Avg Response Time: {metrics['response_time']:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to get system status: {e}")
        return False

def run_comprehensive_test():
    """Run a comprehensive test of the entire system."""
    print("\n🧪 Running Comprehensive System Test...")
    
    test_queries = [
        ("Legal/Policy", "What are the legal requirements for children's home inspections?"),
        ("Technical", "How do I configure error logging in a Python application?"),
        ("General", "What is the purpose of safeguarding policies?"),
        ("Analytical", "Compare different approaches to child protection in residential care.")
    ]
    
    try:
        from rag_system import EnhancedRAGSystem
        
        rag = EnhancedRAGSystem()
        results = []
        
        for category, query in test_queries:
            print(f"  🔍 Testing {category}: '{query[:50]}...'")
            
            start_time = time.time()
            result = rag.query(query, k=3)
            end_time = time.time()
            
            test_result = {
                "category": category,
                "query": query,
                "success": result.get("success", True),
                "response_time": end_time - start_time,
                "confidence": result.get("confidence_score", 0),
                "provider": result.get("routing_info", {}).get("embedding_provider", "unknown"),
                "sources": result.get("total_sources", 0)
            }
            
            results.append(test_result)
            
            status = "✅" if test_result["success"] else "❌"
            print(f"    {status} {test_result['provider']} | "
                  f"Confidence: {test_result['confidence']:.2f} | "
                  f"Time: {test_result['response_time']:.2f}s | "
                  f"Sources: {test_result['sources']}")
        
        # Summary
        successful_tests = sum(1 for r in results if r["success"])
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        avg_time = sum(r["response_time"] for r in results) / len(results)
        
        print(f"\n  📊 Test Summary:")
        print(f"    Success Rate: {successful_tests}/{len(results)} ({(successful_tests/len(results)*100):.1f}%)")
        print(f"    Average Confidence: {avg_confidence:.2f}")
        print(f"    Average Response Time: {avg_time:.2f}s")
        
        # Provider distribution
        provider_usage = {}
        for result in results:
            provider = result["provider"]
            provider_usage[provider] = provider_usage.get(provider, 0) + 1
        
        print(f"    Provider Usage:")
        for provider, count in provider_usage.items():
            print(f"      {provider}: {count}/{len(results)} queries")
        
        return successful_tests == len(results)
        
    except Exception as e:
        print(f"❌ Comprehensive test failed: {e}")
        return False

def main():
    """Main setup and test function."""
    print("🚀 Smart Index Selection RAG System - Setup & Test")
    print("=" * 60)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Check indexes
    indexes_ok, available_indexes = check_indexes()
    if not indexes_ok:
        print("\n💡 To create indexes, run:")
        print("   python enhanced_ingest_dual_indexes.py")
        sys.exit(1)
    
    print(f"\n🎯 Available indexes: {', '.join(available_indexes)}")
    
    # Test components
    tests = [
        ("Smart Router", test_smart_router),
        ("RAG System", test_rag_system),
        ("Embedding Comparison", test_embedding_comparison),
        ("System Status", show_system_status),
        ("Comprehensive Test", run_comprehensive_test)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    # Final summary
    print("\n" + "="*60)
    print("🏁 SETUP AND TEST SUMMARY")
    print("="*60)
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Your Smart Index Selection RAG system is ready!")
        print("\n💡 Next steps:")
        print("  1. Try the example usage in enhanced_rag_system.py")
        print("  2. Integrate the system into your application")
        print("  3. Monitor performance metrics in the performance_metrics/ directory")
        print("  4. Customize routing rules in smart_query_router.py if needed")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("   Common issues:")
        print("   - Missing or incomplete indexes")
        print("   - API keys not properly set")
        print("   - Missing dependencies")
    
    print("\n📚 Quick Usage Example:")
    print("```python")
    print("from enhanced_rag_system import create_rag_system")
    print("")
    print("# Create RAG system")
    print("rag = create_rag_system()")
    print("")
    print("# Ask a question")
    print("result = rag.query('What are safeguarding policies for children?')")
    print("print(result['answer'])")
    print("```")

if __name__ == "__main__":
    main()
