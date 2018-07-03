from telluric.context import local_context, TelluricContext


def test_context_in_one_level():
    with TelluricContext(a=1, b=2, c='stam', d={'a': 'a', 'b': 'b'}):
        assert local_context.get('a') == 1
        assert local_context.get('b') == 2
        assert local_context.get('c') == 'stam'
        assert local_context.get('d') == {'a': 'a', 'b': 'b'}
    assert local_context._options is None


def test_context_in_two_level():
    with TelluricContext(a=1, b=2, c='stam', d={'a': 'a', 'b': 'b'}):
        with TelluricContext(a=4, b=5, x='something', y=13):
            assert local_context.get('a') == 4
            assert local_context.get('b') == 5
            assert local_context.get('c') == 'stam'
            assert local_context.get('d') == {'a': 'a', 'b': 'b'}
            assert local_context.get('x') == 'something'
            assert local_context.get('y') == 13
        assert local_context.get('a') == 1
        assert local_context.get('b') == 2
        assert local_context.get('c') == 'stam'
        assert local_context.get('d') == {'a': 'a', 'b': 'b'}
    assert local_context._options is None


def test_different_context_on_different_threads():
    import threading

    class MyThread (threading.Thread):
        def start(self):
            with TelluricContext(a=1, b=2, c='stam', d={'a': 'a', 'b': 'b'}):
                assert local_context.get('a') == 1
                assert local_context.get('b') == 2
                assert local_context.get('c') == 'stam'
                assert local_context.get('d') == {'a': 'a', 'b': 'b'}
                assert local_context.get('x') is None
                assert local_context.get('y') is None
            assert local_context._options is None

    class MyThread2 (threading.Thread):
        def start(self):
            with TelluricContext(a=4, b=5, x='something', y=13):
                assert local_context.get('a') == 4
                assert local_context.get('b') == 5
                assert local_context.get('x') == 'something'
                assert local_context.get('y') == 13
                assert local_context.get('c') is None
                assert local_context.get('d') is None
            assert local_context._options is None

    assert local_context._options is None
    MyThread().start()
    assert local_context._options is None
    MyThread2().start()
    assert local_context._options is None
