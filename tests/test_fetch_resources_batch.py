#!/usr/bin/env python3
"""
Standalone test for fetch_resources_batch logic.
Tests the core functionality without requiring the full toolset framework.
"""
import asyncio
import base64
import mimetypes
import os
from pathlib import Path


async def fetch_resources_batch(
    workspace_path: Path,
    resource_paths: list[str],
    base_path: str | None = None,
) -> dict:
    """
    Simplified version of fetch_resources_batch for testing.
    This is the core logic extracted from the FileManagerToolSet method.
    """
    results = []
    loaded_count = 0
    failed_count = 0
    
    for resource_path in resource_paths:
        result = {
            "path": resource_path,
            "success": False
        }
        
        try:
            # Resolve path
            if os.path.isabs(resource_path):
                # Absolute path
                target_path = Path(resource_path)
            elif base_path:
                # Relative path with base_path
                base = Path(base_path) if os.path.isabs(base_path) else workspace_path / base_path
                target_path = (base / resource_path).resolve()
            else:
                # Relative path without base_path (relative to workspace)
                target_path = workspace_path / resource_path
            
            result["resolved_path"] = str(target_path)
            
            # Validate path exists
            if not target_path.exists():
                result["error"] = "Resource file does not exist"
                failed_count += 1
                results.append(result)
                continue
            
            if not target_path.is_file():
                result["error"] = "Path is not a file"
                failed_count += 1
                results.append(result)
                continue
            
            # Determine MIME type
            mime_type, _ = mimetypes.guess_type(str(target_path))
            if mime_type is None:
                mime_type = "application/octet-stream"
            
            result["mime_type"] = mime_type
            
            # Load resource based on type
            if mime_type.startswith("image/"):
                # Return base64 data URI for images
                with open(target_path, "rb") as f:
                    file_bytes = f.read()
                content = base64.b64encode(file_bytes).decode()
                result["content"] = f"data:{mime_type};base64,{content}"
            else:
                # Return text content for CSS/JS/HTML
                try:
                    with open(target_path, "r", encoding="utf-8") as f:
                        result["content"] = f.read()
                except UnicodeDecodeError:
                    result["error"] = "File is not a valid text file"
                    failed_count += 1
                    results.append(result)
                    continue
            
            result["success"] = True
            loaded_count += 1
            
        except Exception as e:
            result["error"] = str(e)
            failed_count += 1
        
        results.append(result)
    
    return {
        "success": True,
        "resources": results,
        "total": len(resource_paths),
        "loaded": loaded_count,
        "failed": failed_count
    }


async def run_tests():
    """Run comprehensive tests for fetch_resources_batch."""
    
    # Create test workspace
    test_dir = Path(__file__).parent / "test_workspace_standalone"
    test_dir.mkdir(exist_ok=True)
    
    # Create test directory structure
    (test_dir / "pages").mkdir(exist_ok=True)
    (test_dir / "images").mkdir(exist_ok=True)
    (test_dir / "styles").mkdir(exist_ok=True)
    (test_dir / "scripts").mkdir(exist_ok=True)
    
    print("📁 Creating test files...")
    
    # Create a simple 1x1 red pixel PNG
    image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01\x00\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    (test_dir / "images" / "logo.png").write_bytes(image_data)
    print("  ✓ Created images/logo.png")
    
    # Create CSS file
    css_content = "body { font-family: Arial; }\nh1 { color: #333; }"
    (test_dir / "styles" / "main.css").write_text(css_content)
    print("  ✓ Created styles/main.css")
    
    # Create JS file
    js_content = "console.log('Hello');\nfunction greet() { return 'Hi'; }"
    (test_dir / "scripts" / "app.js").write_text(js_content)
    print("  ✓ Created scripts/app.js\n")
    
    # TEST 1: Relative paths with base_path
    print("=" * 70)
    print("TEST 1: Batch fetch with relative paths (../)")
    print("=" * 70)
    
    result1 = await fetch_resources_batch(
        workspace_path=test_dir,
        resource_paths=["../images/logo.png", "../styles/main.css", "../scripts/app.js"],
        base_path="pages"
    )
    
    print(f"\n📊 Results: Total={result1['total']}, Loaded={result1['loaded']}, Failed={result1['failed']}\n")
    
    for res in result1['resources']:
        status = "✅" if res['success'] else "❌"
        print(f"{status} {res['path']}")
        print(f"   Resolved: {res.get('resolved_path', 'N/A')}")
        if res['success']:
            print(f"   MIME: {res.get('mime_type')}")
            content = res['content']
            if content.startswith('data:'):
                print(f"   Content: {content[:60]}... (base64 data URI)")
            else:
                print(f"   Content: {content[:60]}...")
        else:
            print(f"   Error: {res.get('error')}")
        print()
    
    assert result1['loaded'] == 3, f"Expected 3 loaded, got {result1['loaded']}"
    assert result1['failed'] == 0, f"Expected 0 failed, got {result1['failed']}"
    print("✓ Test 1 PASSED\n")
    
    # TEST 2: Absolute paths
    print("=" * 70)
    print("TEST 2: Batch fetch with absolute paths")
    print("=" * 70)
    
    result2 = await fetch_resources_batch(
        workspace_path=test_dir,
        resource_paths=[
            str(test_dir / "images" / "logo.png"),
            str(test_dir / "styles" / "main.css")
        ]
    )
    
    print(f"\n📊 Results: Total={result2['total']}, Loaded={result2['loaded']}, Failed={result2['failed']}\n")
    
    for res in result2['resources']:
        status = "✅" if res['success'] else "❌"
        print(f"{status} {res['path']}")
        if res['success']:
            print(f"   MIME: {res.get('mime_type')}")
        print()
    
    assert result2['loaded'] == 2, f"Expected 2 loaded, got {result2['loaded']}"
    print("✓ Test 2 PASSED\n")
    
    # TEST 3: Error handling - missing files
    print("=" * 70)
    print("TEST 3: Error handling - non-existent files")
    print("=" * 70)
    
    result3 = await fetch_resources_batch(
        workspace_path=test_dir,
        resource_paths=[
            "../images/logo.png",      # exists
            "../images/missing.png",   # doesn't exist
            "../styles/main.css",      # exists
            "../styles/missing.css"    # doesn't exist
        ],
        base_path="pages"
    )
    
    print(f"\n📊 Results: Total={result3['total']}, Loaded={result3['loaded']}, Failed={result3['failed']}\n")
    
    for res in result3['resources']:
        status = "✅" if res['success'] else "❌"
        print(f"{status} {res['path']}")
        if not res['success']:
            print(f"   Error: {res.get('error')}")
        print()
    
    assert result3['loaded'] == 2, f"Expected 2 loaded, got {result3['loaded']}"
    assert result3['failed'] == 2, f"Expected 2 failed, got {result3['failed']}"
    print("✓ Test 3 PASSED\n")
    
    # TEST 4: Verify image base64 encoding
    print("=" * 70)
    print("TEST 4: Verify image base64 data URI format")
    print("=" * 70)
    
    result4 = await fetch_resources_batch(
        workspace_path=test_dir,
        resource_paths=["images/logo.png"]
    )
    
    img_resource = result4['resources'][0]
    assert img_resource['success'], "Image should load successfully"
    assert img_resource['content'].startswith("data:image/png;base64,"), "Should be base64 data URI"
    
    print(f"✅ Image loaded as data URI")
    print(f"   Format: {img_resource['content'][:40]}...")
    print(f"   Length: {len(img_resource['content'])} chars")
    print("\n✓ Test 4 PASSED\n")
    
    # TEST 5: Verify text file content
    print("=" * 70)
    print("TEST 5: Verify CSS/JS text content")
    print("=" * 70)
    
    result5 = await fetch_resources_batch(
        workspace_path=test_dir,
        resource_paths=["styles/main.css", "scripts/app.js"]
    )
    
    css_res = result5['resources'][0]
    js_res = result5['resources'][1]
    
    assert css_res['success'], "CSS should load"
    assert js_res['success'], "JS should load"
    assert "font-family" in css_res['content'], "CSS content should be correct"
    assert "console.log" in js_res['content'], "JS content should be correct"
    
    print(f"✅ CSS content: {css_res['content'][:50]}...")
    print(f"✅ JS content: {js_res['content'][:50]}...")
    print("\n✓ Test 5 PASSED\n")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    print("🧹 Cleaned up test workspace\n")
    
    print("=" * 70)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 70)
    print("\nSummary:")
    print("  ✓ Relative path resolution with base_path")
    print("  ✓ Absolute path handling")
    print("  ✓ Error handling for missing files")
    print("  ✓ Image base64 encoding")
    print("  ✓ Text file content loading")


if __name__ == "__main__":
    asyncio.run(run_tests())
